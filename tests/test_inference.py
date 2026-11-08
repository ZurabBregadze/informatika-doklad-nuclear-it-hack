"""Submission contract, frozen inference and compatibility of exported weights."""
import hashlib
import json
from pathlib import Path
import struct
import stat
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from doe_p0.inference import discover_inputs, read_inputs, run
from doe_p0.model import FresnelNetwork
from doe_p0.phase import encode_phase
from doe_p0.weights import export_weights, load_weights


@pytest.fixture
def weights(tmp_path):
    model = FresnelNetwork(width=4, feedback_steps=1).eval()
    source = tmp_path/'training.pt'
    torch.save(dict(config=dict(width=4, feedback_steps=1, architecture='joint_spectral_v2'),
                    model=model.state_dict(), epoch=3, selection_criterion='validation_ssim'), source)
    target = tmp_path/'weights/model.pt'
    export_weights(source, target)
    return target


def make_set(folder, index=0):
    folder.mkdir(parents=True)
    for k in range(1, 5):
        array = np.zeros((128, 128), dtype=np.uint8)
        array[8*k:8*k+10+index, 7*k:7*k+11] = 150+index
        Image.fromarray(array).save(folder/f'image_{k}.png')
    return discover_inputs(folder)


def test_export_preserves_output_and_has_no_optimizer(weights, tmp_path):
    original, _ = load_weights(tmp_path/'training.pt')
    restored, saved = load_weights(weights)
    assert saved['export_kind']=='inference' and 'optimizer' not in saved
    inputs = torch.rand(1, 4, 128, 128)
    with torch.inference_mode():
        assert torch.equal(original(inputs), restored(inputs))
    metadata = json.loads(weights.with_suffix('.json').read_text())
    assert metadata['sha256']==hashlib.sha256(weights.read_bytes()).hexdigest()


def test_five_sets_one_model_load_and_exact_output_contract(weights, tmp_path, monkeypatch):
    import doe_p0.inference as inference
    original_loader = inference.load_weights
    loaded = []
    def counted_load(*args):
        loaded.append(1)
        return original_loader(*args)
    monkeypatch.setattr(inference, 'load_weights', counted_load)
    tasks = [dict(sample=f'set_{k}', inputs=make_set(tmp_path/f'input/set_{k}', k),
                  output=tmp_path/f'output/set_{k}/doe_phase.png') for k in range(5)]
    report = run(tasks, weights, device='cpu', threads=1, report_path=tmp_path/'output/run.json')
    assert len(loaded)==1 and len(report['scenes'])==5
    for task in tasks:
        path = task['output']
        assert struct.unpack('>IIBB', path.read_bytes()[16:26])==(128, 128, 8, 0)
        assert stat.S_IMODE(path.stat().st_mode)==0o644
        with Image.open(path) as image:
            assert image.mode=='L' and image.size==(128, 128)
    assert json.loads((tmp_path/'output/run.json').read_text())==report


def test_cli_matches_direct_model_inference_without_training_imports(weights, tmp_path):
    root = Path(__file__).resolve().parents[1]
    inputs = make_set(tmp_path/'input')
    output = tmp_path/'doe.png'
    result = subprocess.run([sys.executable, str(root/'infer.py'), '--input-dir', str(tmp_path/'input'),
                             '--output', str(output), '--checkpoint', str(weights), '--device', 'cpu', '--threads', '1'],
                            check=True, capture_output=True, text=True, cwd=tmp_path)
    assert json.loads(result.stdout)['checkpoint_epoch']==3
    model, _ = load_weights(weights)
    with torch.inference_mode():
        expected = encode_phase(torch.angle(model(read_inputs(inputs)[None]))[0])
    with Image.open(output) as image:
        assert np.array_equal(np.array(image), expected.numpy())
    code = "import sys; from doe_p0.inference import main; assert 'doe_p0.training' not in sys.modules; assert 'doe_p0.fresnel_baseline' not in sys.modules; assert 'matplotlib' not in sys.modules"
    import os
    subprocess.run([sys.executable, '-c', code], check=True,
                   env={**os.environ, 'PYTHONPATH':str(root/'src')}, cwd=tmp_path)


def test_input_modes_and_order_preserve_training_grayscale(tmp_path):
    paths = make_set(tmp_path/'data')
    inputs = read_inputs(paths)
    for k, path in enumerate(paths):
        with Image.open(path) as image:
            assert np.array_equal(inputs[k].numpy(), np.array(image, dtype=np.float32)/255)
            image.convert('RGB').save(tmp_path/f'rgb_{k}.png')
    assert torch.equal(inputs, read_inputs([tmp_path/f'rgb_{k}.png' for k in range(4)]))
    Image.fromarray(np.zeros((128, 128), dtype=np.uint8)).save(paths[0])
    assert read_inputs(paths)[0].max()==0  # Black inputs are not prohibited by the submission rules.
    Image.new('L', (127, 128)).save(paths[0])
    with pytest.raises(ValueError, match='128x128'):
        read_inputs(paths)


def test_checksum_and_input_overwrite_are_rejected(weights, tmp_path):
    inputs = make_set(tmp_path/'data')
    with pytest.raises(ValueError, match='overwrite'):
        run([dict(sample='s', inputs=inputs, output=inputs[0])], weights, threads=1)
    metadata = json.loads(weights.with_suffix('.json').read_text())
    metadata['sha256']='incorrect'
    weights.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='checksum'):
        run([dict(sample='s', inputs=inputs, output=tmp_path/'out.png')], weights, threads=1)


def test_ambiguous_plane_order_is_not_guessed(tmp_path):
    paths = make_set(tmp_path/'data')
    for k, path in enumerate(paths, 1):
        (path.parent/f'{k}.png').write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match='ambiguous'):
        discover_inputs(paths[0].parent)
