"""Differentiable amplitude loss with global PCC and SSIM."""
import torch


def normalize(images):
    return images / images.amax((-2, -1), keepdim=True).clamp_min(1e-8)


def correlations(a, b):
    ac = a - a.mean((-2, -1), keepdim=True)
    bc = b - b.mean((-2, -1), keepdim=True)
    va = ac.square().mean((-2, -1))
    vb = bc.square().mean((-2, -1))
    covariance = (ac * bc).mean((-2, -1))
    denominator = va.clamp_min(1e-12).sqrt() * vb.clamp_min(1e-12).sqrt()
    pcc = (covariance / denominator).clamp(-1, 1)
    return pcc, va, vb, covariance


def global_ssim(a, b):
    _, va, vb, covariance = correlations(a, b)
    ma, mb = a.mean((-2, -1)), b.mean((-2, -1))
    return ((2*ma*mb + .01**2) * (2*covariance + .03**2)
            / ((ma.square() + mb.square() + .01**2) * (va + vb + .03**2)))


def physical_loss(amplitudes, targets, *, return_components=False):
    predicted, expected = normalize(amplitudes), normalize(targets)
    pcc, _, _, _ = correlations(predicted, expected)
    variable = expected.amax((-2, -1)) > expected.amin((-2, -1))
    pcc_loss = torch.where(variable, 1-pcc, 0.)
    unit_targets = expected / expected.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
    amplitude_loss = (amplitudes-unit_targets).square().mean((-2, -1))
    ssim = global_ssim(predicted, expected)
    ssim_loss = 1-ssim
    value = (pcc_loss + 1-ssim + .1*amplitude_loss).mean()
    if not torch.isfinite(value):
        raise FloatingPointError('Physical loss is nonfinite.')
    if return_components:
        return value, dict(pcc_loss=pcc_loss.mean(),
                           ssim_loss=ssim_loss.mean(),
                           amplitude_mse=amplitude_loss.mean())
    return value
