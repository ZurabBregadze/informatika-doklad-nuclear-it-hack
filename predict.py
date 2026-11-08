"""Export one neural DOE and reconstructions for four user PNGs."""
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT/'src'))

from doe_p0.data import read_images
from doe_p0.metrics import export_scene
from doe_p0.training import load_model, seed_everything, select_device

CHECKPOINT = PROJECT/'weights'/'model.pt'
INPUT = [PROJECT/'examples/sets/sample_0451'/f'image_{k}.png' for k in range(1, 5)]
OUTPUT = PROJECT/'predictions'/'example'
DEVICE = 'auto'

if __name__ == '__main__':
    torch.set_num_threads(4)
    seed_everything(0)
    device = select_device(DEVICE)
    model, _ = load_model(CHECKPOINT, device)
    targets = read_images(INPUT)
    with torch.no_grad():
        phase = torch.angle(model(targets[None].to(device)))[0]
    print(export_scene(phase, targets, OUTPUT))
