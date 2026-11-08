"""One shared PNG protocol for neural predictions, validation and GS."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .fresnel_baseline import encode_phase, decode_phase, image_metrics, amplitude_codes
from .model import Optics, DISTANCES


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@torch.no_grad()
def quantized_reconstruction(phase, optics):
    """In-memory equivalent of the exported mask and reconstruction PNG codes."""
    phase = decode_phase(encode_phase(phase.detach().cpu()))
    return amplitude_codes(optics(torch.exp(1j*phase)).abs())


@torch.no_grad()
def quantized_metrics(phase, targets, optics, *, return_images=False):
    """Evaluate one scene using the same codes as the physical PNG round trip."""
    codes = quantized_reconstruction(phase, optics)
    planes = []
    targets = torch.round(targets.detach().cpu().double()*255)/255
    for k, (target, restored) in enumerate(zip(targets, codes), 1):
        target = target / target.max()
        values = image_metrics(target, restored.double()/255)
        planes.append(dict(plane=k, distance_mm=DISTANCES[k-1]*1000,
                           pcc=values['pearson'], ssim=values['ssim_global']))
    result = summarize(planes)
    return (result, codes) if return_images else result


def summarize(planes):
    pcc = [plane['pcc'] for plane in planes]
    return dict(planes=planes, mean_pcc=None if None in pcc else sum(pcc)/len(pcc),
                mean_ssim=sum(plane['ssim'] for plane in planes)/len(planes))


@torch.no_grad()
def export_scene(phase, targets, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    codes = encode_phase(phase.detach().cpu())
    Image.fromarray(codes.numpy()).save(root/'doe_phase.png')
    with Image.open(root/'doe_phase.png') as image:
        decoded = decode_phase(torch.from_numpy(np.array(image)))
    optics = Optics(dtype=torch.float64)
    amplitudes = optics(torch.exp(1j*decoded)).abs()
    targets = torch.round(targets.detach().cpu().double()*255)/255
    planes = []
    for k, (target, amplitude) in enumerate(zip(targets, amplitudes), 1):
        Image.fromarray(amplitude_codes(amplitude).numpy()).save(root/f'reconstruction_{k}.png')
        Image.fromarray(torch.round(target*255).byte().numpy()).save(root/f'target_{k}.png')
        with Image.open(root/f'reconstruction_{k}.png') as image:
            restored = torch.from_numpy(np.array(image)).double()/255
        target = target/target.max()
        values = image_metrics(target, restored)
        planes.append(dict(plane=k, distance_mm=DISTANCES[k-1]*1000,
                           pcc=values['pearson'], ssim=values['ssim_global']))
        error = torch.round(255*(target-restored).abs()).byte().numpy()
        Image.fromarray(error).save(root/f'error_{k}.png')
    result = summarize(planes)
    write_json(root/'metrics.json', result)
    return result
