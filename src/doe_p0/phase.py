"""Shared 8-bit phase codec; no synthesis, optimization or plotting imports."""
import math

import torch


def encode_phase(phase):
    """Encode radians: -pi -> 0, +pi -> 255, with nearest-integer rounding."""
    if not isinstance(phase, torch.Tensor) or not phase.is_floating_point():
        raise TypeError('phase must be a real floating point tensor')
    if not torch.isfinite(phase).all():
        raise ValueError('phase must be finite')
    inside_interval = (phase >= -math.pi) & (phase <= math.pi)
    precise_phase = phase.to(torch.float64)
    wrapped_phase = torch.angle(torch.exp(1j*precise_phase))
    signed_phase = torch.where(inside_interval, precise_phase, wrapped_phase)
    codes = torch.round((signed_phase+math.pi)*(255/(2*math.pi)))
    return codes.clamp(0, 255).to(torch.uint8)


def decode_phase(encoded):
    if not isinstance(encoded, torch.Tensor) or encoded.dtype != torch.uint8:
        raise TypeError('encoded phase must be a uint8 tensor')
    return encoded.to(torch.float64)*(2*math.pi/255)-math.pi
