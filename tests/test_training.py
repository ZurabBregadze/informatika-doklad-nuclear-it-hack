"""Regression tests for the new model, early stopping and exported metrics."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from doe_p0.data import Scenes, audit_dataset
from doe_p0.model import FourierResidual, FresnelNetwork, Optics
from doe_p0.losses import physical_loss
from doe_p0.metrics import export_scene, quantized_metrics
from doe_p0.training import DEFAULT_CONFIG, EarlyStopping, make_scheduler, train, load_model
from doe_p0.comparison import compare


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path/'data'
    rng = np.random.default_rng(13)
    for split in ('train', 'val', 'test'):
        folder = root/split/'sample_0001'
        folder.mkdir(parents=True)
        for k in range(1, 5):
            Image.fromarray(rng.integers(1, 160, (128, 128), dtype=np.uint8)).save(folder/f'image_{k}.png')
    return root


def test_physics_fields_inverse_and_energy():
    optics = Optics(dtype=torch.float64)
    field = torch.randn(2, 128, 128, dtype=torch.complex128)
    output = optics(field)
    assert output.shape == (2, 4, 128, 128)
    assert torch.allclose(output.abs().square().sum((-2, -1)), field.abs().square().sum((-2, -1))[:, None], rtol=1e-12)
    recovered = optics.backward_fields(output)
    assert torch.allclose(recovered, field[:, None].expand_as(recovered), atol=2e-14)


@pytest.mark.parametrize('architecture', ['independent_v1', 'joint_spectral_v2'])
@pytest.mark.parametrize('steps', [0, 2])
def test_network_and_physical_loss_gradients(steps, architecture):
    model = FresnelNetwork(width=4, feedback_steps=steps, architecture=architecture)
    targets = torch.rand(1, 4, 128, 128)
    stages = model(targets, return_stages=True)
    assert len(stages) == steps+1
    for field in stages:
        assert field.shape == (1, 128, 128)
        assert torch.allclose(field.abs(), torch.ones_like(field.real), atol=5e-7)
    loss = physical_loss(model.optics(stages[-1]).abs(), targets)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.field_encoder.head.weight.grad.abs().sum() > 0


def test_joint_encoder_uses_every_target_for_each_predicted_field():
    torch.manual_seed(14)
    model = FresnelNetwork(width=4, feedback_steps=0)
    targets = torch.rand(1, 4, 128, 128, requires_grad=True)
    predicted = model.field_encoder(targets).reshape(1, 4, 2, 128, 128)
    for plane in range(4):
        gradient, = torch.autograd.grad(predicted[:, plane].square().mean(), targets, retain_graph=True)
        # Changing any target can influence any one of the four complex fields.
        assert (gradient.abs().sum((-2, -1)) > 0).all()


def test_fourier_branch_gradients_and_global_connection():
    torch.manual_seed(15)
    branch = FourierResidual(2, modes=2).double()
    values = torch.randn(1, 2, 8, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(branch, (values,), fast_mode=True)
    gradient, = torch.autograd.grad(branch(values)[0, 0, 0, 0], values)
    # The distant pixel lies outside the pointwise projections' receptive field.
    assert gradient[0, :, 4, 4].abs().sum() > 0


@pytest.mark.parametrize('architecture', ['independent_v1', 'joint_spectral_v2'])
def test_checkpoint_preserves_architecture_and_output(tmp_path, architecture):
    config = dict(width=4, feedback_steps=1)
    if architecture != 'independent_v1':
        config['architecture'] = architecture
    model = FresnelNetwork(width=4, feedback_steps=1, architecture=architecture)
    model.eval()
    inputs = torch.rand(1, 4, 128, 128)
    checkpoint = tmp_path/'model.pt'
    torch.save(dict(config=config, model=model.state_dict()), checkpoint)
    restored, _ = load_model(checkpoint, 'cpu')
    assert restored.architecture == architecture
    with torch.no_grad():
        assert torch.equal(model(inputs), restored(inputs))


def test_early_stop_ten_epochs_and_accumulated_small_improvements():
    stopper = EarlyStopping()
    assert not stopper.update(1.)
    for _ in range(9):
        assert not stopper.update(1.01)
    assert stopper.update(1.01)
    stopper = EarlyStopping(min_delta=.001)
    assert not stopper.update(1.)
    assert not stopper.update(.9994)
    assert stopper.bad_epochs == 1
    assert not stopper.update(.9988)
    assert stopper.bad_epochs == 0


def test_png_metric_roundtrip(dataset, tmp_path):
    targets = Scenes(dataset, 'test')[0]
    phase = torch.randn(128, 128).clamp(-3, 3)
    result = export_scene(phase, targets, tmp_path/'scene')
    expected = quantized_metrics(phase, targets, Optics(dtype=torch.float64))
    assert result == expected
    for k in range(1, 5):
        with Image.open(tmp_path/f'scene/reconstruction_{k}.png') as image:
            assert image.mode == 'L' and image.size == (128, 128)
    measured, codes = quantized_metrics(phase, targets, Optics(dtype=torch.float64), return_images=True)
    assert measured == result
    for k in range(4):
        with Image.open(tmp_path/f'scene/reconstruction_{k+1}.png') as image:
            assert np.array_equal(codes[k].numpy(), np.array(image))


def test_loss_components_preserve_value_and_gradients():
    amplitudes = torch.rand(2, 4, 16, 16, requires_grad=True)
    targets = torch.rand_like(amplitudes)
    expected = physical_loss(amplitudes, targets)
    value, terms = physical_loss(amplitudes, targets, return_components=True)
    assert torch.equal(value, expected)
    assert torch.allclose(value, terms['pcc_loss']+terms['ssim_loss']+.1*terms['amplitude_mse'])
    expected_gradient, = torch.autograd.grad(expected, amplitudes)
    actual_gradient, = torch.autograd.grad(value, amplitudes)
    assert torch.equal(expected_gradient, actual_gradient)


def test_lr_reductions_do_not_reset_early_stopping():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
    scheduler = make_scheduler(optimizer, DEFAULT_CONFIG)
    stopper = EarlyStopping()
    for epoch in range(11):
        loss = 1. if epoch == 0 else 1.01
        scheduler.step(loss)
        stopped = stopper.update(loss)
        assert stopped == (epoch == 10)
        if epoch == 3:
            assert optimizer.param_groups[0]['lr'] == 5e-5
            assert stopper.bad_epochs == 3
    assert optimizer.param_groups[0]['lr'] == 1.25e-5


def test_short_fixture_train_resume_and_gs_comparison(dataset, tmp_path):
    # Exactly one tiny fixture update per invocation. No real-dataset training.
    config = dict(width=4, feedback_steps=1, batch_size=1, max_epochs=1, threads=1, device='cpu')
    root = train(dataset, tmp_path/'run', config, backup_dir=tmp_path/'backups')
    state = torch.load(root/'last.pt', weights_only=True)
    assert state['epoch'] == 1 and state['global_step'] == 1
    assert state['scheduler'] is not None
    row = state['history'][0]
    assert row['train_loss'] == pytest.approx(row['train_final_loss']+.2*row['train_aux_loss'])
    assert row['train_final_loss'] == pytest.approx(row['train_pcc_loss']+row['train_ssim_loss']+.1*row['train_amplitude_mse'])
    assert row['val_loss'] == pytest.approx(row['val_pcc_loss']+row['val_ssim_loss']+.1*row['val_amplitude_mse'])
    assert row['gradient_norm_mean'] > 0
    assert row['learning_rate'] == 1e-4
    from doe_p0.backup import restore_run
    restored_path = restore_run(tmp_path/'backups', tmp_path/'restored')
    restored_state = torch.load(restored_path, weights_only=True)
    assert restored_state['global_step'] == state['global_step']
    assert torch.equal(restored_state['torch_rng'], state['torch_rng'])
    for name in ('best_pcc.pt', 'best_ssim.pt', 'stage_history.csv', 'diagnostics.json', 'validation_examples.png'):
        assert (tmp_path/'restored'/name).read_bytes() == (root/name).read_bytes()
    diagnostic = json.loads((root/'diagnostics.json').read_text())
    assert diagnostic['split'] == 'val'
    assert diagnostic['sample_ids'] == ['sample_0001']
    assert [row['stage'] for row in diagnostic['stages']] == [0, 1]
    assert json.loads((root/'status.json').read_text())['stage'] == 'complete'
    train(dataset, root, config, resume=root/'last.pt')
    again = torch.load(root/'last.pt', weights_only=True)
    assert again['global_step'] == state['global_step']
    model, _ = load_model(root/'best.pt', 'cpu')
    assert model(Scenes(dataset, 'train')[0][None]).shape == (1, 128, 128)
    result = compare(root/'best.pt', dataset, tmp_path/'comparison', gs_iterations=2, device='cpu')
    assert result['neural']['samples'] == result['gs']['samples'] == 1
    assert (tmp_path/'comparison/sample_0001/comparison.png').exists()
    timing_path = tmp_path/'comparison/timings.json'
    timing = json.loads(timing_path.read_text())
    assert set(timing['methods']) == {'neural_cpu', 'gs_cpu'}
    for method in timing['methods'].values():
        assert method['core']['count'] == method['pipeline']['count'] == 3
        assert method['core']['median_seconds'] > 0
        assert method['pipeline']['median_seconds'] > 0
    original_timing = timing_path.read_bytes()
    repeated = compare(root/'best.pt', dataset, tmp_path/'comparison', gs_iterations=2, device='cpu')
    assert repeated == result
    assert timing_path.read_bytes() == original_timing
    rows = list(csv.DictReader((tmp_path/'comparison/metrics.csv').open()))
    assert len(rows) == 8
    assert {row['method'] for row in rows} == {'neural', 'gs'}


def test_v2_resume_matches_uninterrupted_fixture_updates(dataset, tmp_path, monkeypatch):
    import doe_p0.training as training
    original_validate = training.validate
    def constant_loss(*args):
        return dict(original_validate(*args), loss=1.)
    monkeypatch.setattr(training, 'validate', constant_loss)
    config = dict(width=4, feedback_steps=1, batch_size=1, max_epochs=3, threads=1,
                  device='cpu', lr_patience=0)
    whole = train(dataset, tmp_path/'whole', config)
    partial = train(dataset, tmp_path/'partial', {**config, 'max_epochs':1})
    checkpoint = torch.load(partial/'last.pt', weights_only=True)
    # Extend only this synthetic test's limit, so it can exercise one update
    # after restoring AdamW, RNG and the loader generator from disk.
    checkpoint['config']['max_epochs'] = 3
    torch.save(checkpoint, partial/'last.pt')
    train(dataset, partial, resume=partial/'last.pt')
    expected = torch.load(whole/'last.pt', weights_only=True)
    actual = torch.load(partial/'last.pt', weights_only=True)
    assert actual['global_step'] == expected['global_step'] == 3
    assert actual['history'] == expected['history']
    assert actual['stage_history'] == expected['stage_history']
    assert actual['scheduler'] == expected['scheduler']
    assert actual['optimizer']['param_groups'][0]['lr'] == 2.5e-5
    assert actual['history'][-1]['learning_rate'] == 5e-5
    for name in expected['model']:
        assert torch.equal(expected['model'][name], actual['model'][name]), name


def test_best_loss_pcc_and_ssim_can_select_different_epochs(dataset, tmp_path, monkeypatch):
    import doe_p0.training as training
    original_validate = training.validate
    scores = iter([(0.9, .7, .2), (.8, .6, .3), (.85, .65, .4)])
    def measured_scores(*args):
        result = original_validate(*args)
        result['loss'], result['mean_pcc'], result['mean_ssim'] = next(scores)
        return result
    monkeypatch.setattr(training, 'validate', measured_scores)
    monkeypatch.setattr(training, 'plot_history', lambda *args:None)
    root = train(dataset, tmp_path/'run', dict(width=4, feedback_steps=0, max_epochs=3,
                 batch_size=1, threads=1, device='cpu', diagnostic_samples=0))
    for filename, epoch in (('best.pt', 2), ('best_pcc.pt', 1), ('best_ssim.pt', 3), ('last.pt', 3)):
        model, saved = load_model(root/filename, 'cpu')
        assert saved['epoch'] == epoch
    compact = torch.load(root/'best_pcc.pt', weights_only=True)
    assert 'optimizer' not in compact
    assert compact['selection_criterion'] == 'validation_pcc'
    with pytest.raises(ValueError, match='last.pt'):
        train(dataset, tmp_path/'cannot_resume', resume=root/'best_pcc.pt')


def test_old_checkpoint_resume_keeps_constant_lr_and_migrates_history(dataset, tmp_path, monkeypatch):
    import doe_p0.training as training
    monkeypatch.setattr(training, 'plot_history', lambda *args:None)
    config = dict(width=4, feedback_steps=0, max_epochs=1, batch_size=1, threads=1,
                  device='cpu', diagnostic_samples=0)
    root = train(dataset, tmp_path/'run', config)
    saved = torch.load(root/'last.pt', weights_only=True)
    for key in ('lr_scheduler', 'lr_factor', 'lr_patience', 'min_lr', 'diagnostic_samples'):
        saved['config'].pop(key)
    for key in ('scheduler', 'stage_history', 'best_pcc', 'best_ssim', 'metric_selection_start_epoch'):
        saved.pop(key)
    saved['history'] = [{key:saved['history'][0][key] for key in ('epoch', 'train_loss', 'val_loss', 'val_pcc', 'val_ssim')}]
    saved['config']['max_epochs'] = 2
    torch.save(saved, root/'last.pt')
    train(dataset, root, resume=root/'last.pt')
    actual = torch.load(root/'last.pt', weights_only=True)
    assert actual['config']['lr_scheduler'] == 'none' and actual['scheduler'] is None
    assert actual['history'][-1]['learning_rate'] == 1e-4
    assert actual['metric_selection_start_epoch'] == 2
    assert len(list(csv.DictReader((root/'history.csv').open()))) == 2


def test_split_leakage_rejected(dataset):
    audit = audit_dataset(dataset)
    assert audit['samples'] == dict(train=1, val=1, test=1)
    source = dataset/'train/sample_0001/image_1.png'
    target = dataset/'val/sample_0001/image_1.png'
    target.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match='across splits'):
        audit_dataset(dataset)


def test_backup_survives_corrupt_latest_slot(tmp_path):
    from doe_p0.backup import backup_run, restore_run
    run = tmp_path/'run'
    run.mkdir()
    for name in ('last.pt', 'best.pt', 'config.json'):
        (run/name).write_text('epoch one')
    backup_run(run, tmp_path/'backup', epoch=1)
    (run/'last.pt').write_text('epoch two')
    backup_run(run, tmp_path/'backup', epoch=2)
    (tmp_path/'backup/checkpoint_0.zip').write_bytes(b'interrupted upload')
    restored = restore_run(tmp_path/'backup', tmp_path/'restored')
    assert restored.read_text() == 'epoch one'
