"""Load and export inference weights independently of training and GS."""
import hashlib
import json
from pathlib import Path

import torch

from .model import model_from_config


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_weights(path, device='cpu'):
    path = Path(path)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(saved, dict) or not {'config', 'model'}.issubset(saved):
        raise ValueError('Expected a checkpoint with config and model state.')
    model = model_from_config(saved['config'])
    model.load_state_dict(saved['model'], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Checkpoint contains nonfinite parameters or optical buffers.')
    return model.to(device).eval(), saved


def export_weights(checkpoint, output):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.suffix != '.pt':
        raise ValueError('Export path must end in .pt.')
    if checkpoint == output:
        raise ValueError('Export to a separate file; keep the training checkpoint intact.')
    model, saved = load_weights(checkpoint)
    with torch.inference_mode():
        field = model(torch.zeros(1, 4, 128, 128))
        if not torch.isfinite(field).all() or not torch.allclose(field.abs(), torch.ones_like(field.real), atol=1e-6):
            raise ValueError('Checkpoint failed the phase-only inference check.')
    validation = saved.get('validation')
    if validation is None and saved.get('history'):
        row = saved['history'][-1]
        validation = dict(loss=row['val_loss'], mean_pcc=row['val_pcc'], mean_ssim=row['val_ssim'])
    compact = dict(format=4, export_kind='inference', config=saved['config'], model=model.state_dict(),
                   epoch=saved.get('epoch'), global_step=saved.get('global_step'),
                   dataset_fingerprint=saved.get('dataset_fingerprint'), validation=validation,
                   selection_criterion=saved.get('selection_criterion', 'unspecified'),
                   source_checkpoint_sha256=sha256(checkpoint))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    torch.save(compact, temporary)
    temporary.replace(output)
    metadata = {key:value for key, value in compact.items() if key not in ('model', 'format')}
    metadata.update(filename=output.name, sha256=sha256(output),
                    parameters=sum(p.numel() for p in model.parameters()),
                    phase_encoding='0 -> -pi; 255 -> +pi; nearest integer; 8-bit grayscale PNG')
    temporary_json = output.with_suffix('.json.tmp')
    temporary_json.write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    temporary_json.replace(output.with_suffix('.json'))
    return metadata
