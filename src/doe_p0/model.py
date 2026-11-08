"""Joint neural synthesis with Fresnel transport and spatial/Fourier features."""
import torch
from torch import nn
from torch.nn import functional as F

from .fresnel import FresnelSingleFFT

DISTANCES = (.0241, .0253, .0265, .0277)
ARCHITECTURES = ('independent_v1', 'joint_spectral_v2')


class Optics(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.operators = nn.ModuleList([
            FresnelSingleFFT((128, 128), distance=z, dtype=dtype) for z in DISTANCES
        ])

    def forward(self, field):
        # Remove only the known constant magnitude, preserving unitary scaling.
        return torch.stack([op(field) / op.factor.abs() for op in self.operators], dim=-3)

    def backward_fields(self, fields):
        return torch.stack([
            op.inverse(fields[..., k, :, :] * op.factor.abs())
            for k, op in enumerate(self.operators)
        ], dim=-3)


def unit_phasor(real, imaginary):
    field = torch.complex(real, imaginary)
    if not torch.isfinite(field).all():
        raise FloatingPointError('Nonfinite predicted complex field.')
    safe = torch.where(field.abs() > 1e-7, field, torch.ones_like(field))
    return safe / safe.abs()


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(groups, channels),
        )

    def forward(self, features):
        return F.silu(features + self.layers(features))


class FourierResidual(nn.Module):
    """A learned low-frequency branch alongside the existing spatial features.

    Projection to 16 channels keeps the spectral filters small. Both signs of
    the vertical frequencies are retained; rfft stores horizontal frequencies
    only once. The residual path preserves the original high-frequency detail.
    """
    def __init__(self, channels, modes=8):
        super().__init__()
        hidden = min(16, channels)
        self.modes = modes
        self.reduce = nn.Conv2d(channels, hidden, 1)
        # Real/imaginary pairs work with ordinary real-valued AdamW states.
        self.weights = nn.Parameter(torch.randn(2, hidden, hidden, modes, modes, 2)
                                    / (2*hidden)**.5)
        self.expand = nn.Conv2d(hidden, channels, 1)
        # Start as a small perturbation of the spatial network, with gradients
        # reaching the spectral branch from the very first update.
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), .01))

    def forward(self, features):
        values = self.reduce(features)
        height, width = values.shape[-2:]
        spectrum = torch.fft.rfft2(values, norm='ortho')
        filtered = torch.zeros_like(spectrum)
        rows = min(self.modes, height//2)
        columns = min(self.modes, width//2+1)
        weights = torch.view_as_complex(self.weights.contiguous())
        filtered[:, :, :rows, :columns] = torch.einsum(
            'bixy,ioxy->boxy', spectrum[:, :, :rows, :columns], weights[0, :, :, :rows, :columns])
        filtered[:, :, -rows:, :columns] = torch.einsum(
            'bixy,ioxy->boxy', spectrum[:, :, -rows:, :columns], weights[1, :, :, :rows, :columns])
        global_features = torch.fft.irfft2(filtered, s=(height, width), norm='ortho')
        return features + self.scale*self.expand(F.silu(global_features))


class FieldNetwork(nn.Module):
    """A small residual encoder/decoder used for complex fields and fusion."""
    def __init__(self, incoming, width, outgoing=2, spectral=False):
        super().__init__()
        self.first = nn.Sequential(nn.Conv2d(incoming, width, 3, padding=1), Residual(width))
        self.down = nn.Sequential(nn.Conv2d(width, 2*width, 3, stride=2, padding=1), Residual(2*width))
        self.bottom = nn.Sequential(nn.Conv2d(2*width, 4*width, 3, stride=2, padding=1), Residual(4*width))
        self.spectral = FourierResidual(4*width) if spectral else nn.Identity()
        self.up_middle = nn.Sequential(nn.Conv2d(6*width, 2*width, 3, padding=1), Residual(2*width))
        self.up_full = nn.Sequential(nn.Conv2d(3*width, width, 3, padding=1), Residual(width))
        self.head = nn.Conv2d(width, outgoing, 1)

    def forward(self, images):
        first = self.first(images)
        middle = self.down(first)
        bottom = self.spectral(self.bottom(middle))
        up = F.interpolate(bottom, size=middle.shape[-2:], mode='bilinear', align_corners=False)
        up = self.up_middle(torch.cat((up, middle), dim=1))
        up = F.interpolate(up, size=first.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(self.up_full(torch.cat((up, first), dim=1)))


class FresnelNetwork(nn.Module):
    def __init__(self, width=32, feedback_steps=2, architecture='joint_spectral_v2'):
        super().__init__()
        if type(width) is not int or width < 2 or type(feedback_steps) is not int or feedback_steps < 0:
            raise ValueError('width >= 2 and feedback_steps >= 0 required.')
        if architecture not in ARCHITECTURES:
            raise ValueError(f'Unknown architecture: {architecture}')
        self.architecture = architecture
        joint = architecture == 'joint_spectral_v2'
        self.feedback_steps = feedback_steps
        self.optics = Optics()
        self.field_encoder = FieldNetwork(4, width, outgoing=8) if joint else FieldNetwork(2, width)
        self.fusion = FieldNetwork(8, width, spectral=joint)
        if feedback_steps:
            self.message_encoder = FieldNetwork(5, width)
            self.phase_update = FieldNetwork(10, width, outgoing=1, spectral=joint)
            nn.init.normal_(self.phase_update.head.weight, std=.01)
            nn.init.zeros_(self.phase_update.head.bias)
        self.register_buffer('depths', torch.linspace(-1, 1, 4).reshape(1, 4, 1, 1))

    @staticmethod
    def complex_channels(fields):
        return torch.cat((fields.real, fields.imag), dim=1)

    def forward(self, targets, return_stages=False):
        if targets.ndim != 4 or tuple(targets.shape[1:]) != (4, 128, 128):
            raise ValueError('Expected [batch, 4, 128, 128].')
        if targets.dtype != torch.float32 or not torch.isfinite(targets).all():
            raise ValueError('Finite float32 amplitudes required.')
        batch = targets.shape[0]
        depth = self.depths.expand_as(targets)
        if self.architecture == 'joint_spectral_v2':
            # Channel order encodes the four fixed physical distances. Each
            # predicted field can depend on all four targets from the outset.
            prediction = self.field_encoder(targets).reshape(batch, 4, 2, 128, 128)
        else:
            inputs = torch.stack((targets, depth), dim=2).reshape(batch*4, 2, 128, 128)
            prediction = self.field_encoder(inputs).reshape(batch, 4, 2, 128, 128)
        amplitude = F.softplus(prediction[:, :, 0]) + 1e-4
        amplitude = amplitude / amplitude.square().mean((-2, -1), keepdim=True).sqrt()
        fields = amplitude * torch.exp(1j * prediction[:, :, 1])
        transported = self.optics.backward_fields(fields)
        components = self.fusion(self.complex_channels(transported))
        field = unit_phasor(components[:, 0], components[:, 1])
        stages = [field]
        for _ in range(self.feedback_steps):
            propagated = self.optics(field)
            amplitude = propagated.abs()
            amplitude = amplitude / amplitude.amax((-2, -1), keepdim=True).clamp_min(1e-8)
            inputs = torch.stack((targets, amplitude, propagated.real, propagated.imag, depth), dim=2)
            messages = self.message_encoder(inputs.reshape(batch*4, 5, 128, 128))
            messages = messages.reshape(batch, 4, 2, 128, 128)
            messages = torch.complex(messages[:, :, 0], messages[:, :, 1])
            messages = self.optics.backward_fields(messages)
            update_inputs = torch.cat((field.real[:, None], field.imag[:, None], self.complex_channels(messages)), dim=1)
            delta = torch.pi * torch.tanh(self.phase_update(update_inputs)[:, 0])
            field = field * torch.exp(1j * delta)
            stages.append(field)
        return stages if return_stages else field


def model_from_config(config):
    """Old checkpoints have no architecture tag and retain their exact graph."""
    return FresnelNetwork(width=config['width'], feedback_steps=config['feedback_steps'],
                          architecture=config.get('architecture', 'independent_v1'))
