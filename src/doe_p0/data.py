"""Read ordered four-plane scenes and audit the train/val/test split."""
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_images(paths):
    images = []
    if len(paths) != 4:
        raise ValueError('Four images are required, in increasing distance order.')
    for path in paths:
        with Image.open(path) as image:
            if (image.format != 'PNG' or image.mode != 'L' or image.size != (128, 128)
                    or 'transparency' in image.info or getattr(image, 'n_frames', 1) != 1):
                raise ValueError(f'Expected opaque grayscale 128x128 PNG: {path}')
            values = np.array(image, dtype=np.float32) / 255
        if values.max() <= 0:
            raise ValueError(f'Empty target: {path}')
        images.append(torch.from_numpy(values))
    return torch.stack(images)


class Scenes(Dataset):
    def __init__(self, root, split):
        self.root = Path(root).resolve()
        if split not in ('train', 'val', 'test'):
            raise ValueError('Expected train, val or test.')
        directory = self.root / split
        self.folders = sorted(p for p in directory.iterdir() if p.is_dir())
        if not self.folders:
            raise ValueError(f'No samples found: {directory}')
        for folder in self.folders:
            expected = {folder / f'image_{k}.png' for k in range(1, 5)}
            if set(folder.glob('*.png')) != expected:
                raise ValueError(f'Expected exactly image_1.png ... image_4.png: {folder}')
            if any(not p.resolve().is_relative_to(self.root) for p in expected):
                raise ValueError('Image path escapes DATA_ROOT.')

    def __len__(self):
        return len(self.folders)

    def paths(self, index):
        return [self.folders[index] / f'image_{k}.png' for k in range(1, 5)]

    def __getitem__(self, index):
        return read_images(self.paths(index))


def audit_dataset(root):
    root = Path(root).resolve()
    fingerprint = hashlib.sha256()
    owners = {}
    paths = set()
    counts = {}
    for split in ('train', 'val', 'test'):
        data = Scenes(root, split)
        counts[split] = len(data)
        for index in range(len(data)):
            values = data[index]
            for path, value in zip(data.paths(index), values):
                relative = path.relative_to(root).as_posix()
                paths.add(relative)
                raw = path.read_bytes()
                fingerprint.update(relative.encode())
                fingerprint.update(hashlib.sha256(raw).digest())
                pixels = torch.round(value * 255).byte().numpy().tobytes()
                owners.setdefault(hashlib.sha256(pixels).hexdigest(), set()).add(split)
    leaks = [key for key, splits in owners.items() if len(splits) > 1]
    if leaks:
        raise ValueError(f'Identical images occur across splits: {len(leaks)}')
    manifest = root / 'manifest.csv'
    if manifest.exists():
        fingerprint.update(manifest.read_bytes())
        rows = list(csv.DictReader(manifest.open(encoding='utf-8-sig')))
        # The small supplied dataset has one manifest row per image.
        if rows and 'file' in rows[0]:
            if len(rows) != len(paths) or {r['file'] for r in rows} != paths:
                raise ValueError('Manifest and actual PNG files disagree.')
            groups = {}
            for row in rows:
                relative = Path(row['file'])
                plane = int(row['plane'])
                if (plane not in (1, 2, 3, 4) or relative.parts[0] != row['split']
                        or relative.name != f'image_{plane}.png'
                        or abs(float(row['z_mm']) - (24.1, 25.3, 26.5, 27.7)[plane-1]) > 1e-6):
                    raise ValueError('Invalid plane/split/distance in manifest.')
                if row.get('file_sha256') and hashlib.sha256((root/relative).read_bytes()).hexdigest() != row['file_sha256']:
                    raise ValueError(f'PNG checksum mismatch: {relative}')
                for key in ('source_uid', 'raw_sha256', 'canonical_sha256', 'raw_group'):
                    value = row.get(key)
                    if value:
                        groups.setdefault((key, value), set()).add(row['split'])
            if any(len(splits) > 1 for splits in groups.values()):
                raise ValueError('Recorded source/group occurs across splits.')
    return dict(samples=counts, images=4*sum(counts.values()), fingerprint=fingerprint.hexdigest(),
                exact_pixel_leaks=0, limitation='Undocumented semantic similarities are not ruled out.')
