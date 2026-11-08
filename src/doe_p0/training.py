"""Training with validation-loss early stopping, 10 stale epochs / 200 maximum."""
import math
import os
import random
import signal
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader

from .backup import backup_run
from .data import Scenes, audit_dataset
from .diagnostics import plot_history, plot_validation, stage_row, write_csv
from .losses import physical_loss
from .metrics import quantized_metrics, write_json
from .model import Optics, model_from_config


DEFAULT_CONFIG = dict(training_revision='2.1', architecture='joint_spectral_v2', width=32, feedback_steps=2, batch_size=4, learning_rate=1e-4,
                      weight_decay=1e-4, max_epochs=200, patience=10, min_delta=1e-5,
                      seed=0, threads=4, device='auto', auxiliary_weight=.2,
                      lr_scheduler='plateau', lr_factor=.5, lr_patience=2, min_lr=1e-6,
                      diagnostic_samples=4)


def select_device(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name not in ('cpu', 'cuda') or (name == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Choose available cpu/cuda. Complex FFT uses CPU on Mac.')
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_checkpoint(path, values):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(values, temporary)
    temporary.replace(path)


def load_model(checkpoint, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = model_from_config(saved['config']).to(device)
    model.load_state_dict(saved['model'])
    return model.eval(), saved


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-5, anchor=math.inf, bad_epochs=0):
        self.patience, self.min_delta = patience, min_delta
        self.anchor, self.bad_epochs = anchor, bad_epochs

    def update(self, loss):
        if not math.isfinite(loss):
            raise FloatingPointError('Nonfinite validation loss.')
        if loss < self.anchor-self.min_delta:
            self.anchor, self.bad_epochs = loss, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def make_scheduler(optimizer, config):
    if config['lr_scheduler'] == 'none':
        return None
    if config['lr_scheduler'] != 'plateau':
        raise ValueError('lr_scheduler must be plateau or none.')
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config['lr_factor'], patience=config['lr_patience'],
        threshold=config['min_delta'], threshold_mode='abs', min_lr=config['min_lr'])


@torch.no_grad()
def validate(model, loader, device, precise, diagnostic_samples=0):
    model.eval()
    total, count, metrics = 0., 0, []
    components = dict(pcc_loss=0., ssim_loss=0., amplitude_mse=0.)
    diagnostics, examples = [], []
    for targets in loader:
        images = targets.to(device)
        stages = model(images, return_stages=True)
        field = stages[-1]
        loss, terms = physical_loss(model.optics(field).abs(), images, return_components=True)
        total += float(loss)*len(images)
        for key, value in terms.items():
            components[key] += float(value)*len(images)
        for index, target in enumerate(targets):
            sample_index = count+index
            if sample_index < diagnostic_samples:
                sample = loader.dataset.folders[sample_index].name
                example = dict(sample=sample, targets=target, reconstructions=[], metrics=[])
                for stage, stage_field in enumerate(stages):
                    result, codes = quantized_metrics(torch.angle(stage_field[index]), target,
                                                      precise, return_images=True)
                    diagnostics.append(stage_row(sample, stage, result))
                    example['reconstructions'].append(codes)
                    example['metrics'].append(result)
                examples.append(example)
                metrics.append(example['metrics'][-1])
            else:
                metrics.append(quantized_metrics(torch.angle(field[index]), target, precise))
        count += len(images)
    pcc = [row['mean_pcc'] for row in metrics]
    return dict(loss=total/count, mean_pcc=None if None in pcc else sum(pcc)/count,
                mean_ssim=sum(row['mean_ssim'] for row in metrics)/count,
                pcc_planes=[None if any(row['planes'][k]['pcc'] is None for row in metrics)
                            else sum(row['planes'][k]['pcc'] for row in metrics)/count for k in range(4)],
                ssim_planes=[sum(row['planes'][k]['ssim'] for row in metrics)/count for k in range(4)],
                components={key:value/count for key, value in components.items()},
                diagnostics=diagnostics, examples=examples)


def train(data_root, output, config=None, resume=None, backup_dir=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Process lock is released by the operating system even after a crash.
    import fcntl
    with (root/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train_locked(data_root, root, config, resume, backup_dir)


def _train_locked(data_root, root, config, resume, backup_dir=None):
    saved = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
    if saved:
        config = dict(saved['config'])
        config.setdefault('architecture', 'independent_v1')
        config.setdefault('training_revision', 'legacy')
        # A checkpoint from before this revision keeps its constant-LR policy.
        config.setdefault('lr_scheduler', 'none')
        for key in ('lr_factor', 'lr_patience', 'min_lr', 'diagnostic_samples'):
            config.setdefault(key, DEFAULT_CONFIG[key])
        if 'optimizer' not in saved:
            raise ValueError('This checkpoint contains inference weights only. Resume from last.pt.')
    elif (root/'config.json').exists():
        raise FileExistsError('Run exists. Set RESUME to its last.pt or choose a new OUTPUT.')
    if not (1 <= config['max_epochs'] <= 200 and 1 <= config['patience'] <= 10
            and config['batch_size'] >= 1 and config['learning_rate'] > 0
            and config['min_delta'] >= 0):
        raise ValueError('Check epochs (1..200), patience (1..10), batch size and learning rate.')
    if config['lr_scheduler'] not in ('plateau', 'none'):
        raise ValueError('lr_scheduler must be plateau or none.')
    if (type(config['diagnostic_samples']) is not int or not 0 <= config['diagnostic_samples'] <= 8
            or (config['lr_scheduler'] == 'plateau' and
                (not 0 < config['lr_factor'] < 1 or type(config['lr_patience']) is not int
                 or config['lr_patience'] < 0 or not 0 < config['min_lr'] <= config['learning_rate']))):
        raise ValueError('Check LR factor/patience/minimum and diagnostic_samples (0..8).')
    device = select_device(config['device'])
    torch.set_num_threads(config['threads'])
    seed_everything(config['seed'])
    audit = audit_dataset(data_root)
    if saved and saved['dataset_fingerprint'] != audit['fingerprint']:
        raise ValueError('Resume dataset differs from the saved dataset.')
    write_json(root/'dataset_audit.json', audit)
    write_json(root/'config.json', config)
    train_data, val_data = Scenes(data_root, 'train'), Scenes(data_root, 'val')
    generator = torch.Generator().manual_seed(config['seed'])
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                              generator=generator, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], num_workers=0)
    model = model_from_config(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = make_scheduler(optimizer, config)
    precise = Optics(dtype=torch.float64)
    stopper = EarlyStopping(config['patience'], config['min_delta'])
    epoch, global_step, best_loss, history, previous_seconds = 0, 0, math.inf, [], 0.
    best_pcc, best_ssim, stage_history, metric_selection_start_epoch = -math.inf, -math.inf, [], 1
    if saved:
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        if scheduler is not None:
            if saved.get('scheduler') is None:
                raise ValueError('Scheduler state is missing. Use the matching last.pt.')
            scheduler.load_state_dict(saved['scheduler'])
        epoch, global_step, best_loss, history = saved['epoch'], saved['global_step'], saved['best_loss'], saved['history']
        stopper.anchor, stopper.bad_epochs = saved['anchor'], saved['bad_epochs']
        generator.set_state(saved['loader_rng'])
        torch.set_rng_state(saved['torch_rng'])
        random.setstate(saved['python_rng'])
        numpy_state = saved['numpy_rng']
        np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
        if device.type == 'cuda' and saved['cuda_rng']:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        previous_seconds = saved['seconds']
        best_pcc, best_ssim = saved.get('best_pcc', -math.inf), saved.get('best_ssim', -math.inf)
        stage_history = saved.get('stage_history', [])
        metric_selection_start_epoch = saved.get('metric_selection_start_epoch', epoch+1)
    print(f"Architecture: {model.architecture}; device: {device}; train/val: {len(train_data)}/{len(val_data)}; parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    requested_stop = False
    handlers = {}
    started = time.perf_counter()

    def request_stop(signum, frame):
        nonlocal requested_stop
        requested_stop = True
        print('Stop requested. Saving at the end of the current epoch.', flush=True)

    def checkpoint():
        numpy_state = np.random.get_state()
        return dict(format=3, config=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict() if scheduler is not None else None,
                    best_pcc=best_pcc, best_ssim=best_ssim, stage_history=stage_history,
                    metric_selection_start_epoch=metric_selection_start_epoch,
                    epoch=epoch, global_step=global_step, best_loss=best_loss, anchor=stopper.anchor,
                    bad_epochs=stopper.bad_epochs, history=history, dataset_fingerprint=audit['fingerprint'],
                    loader_rng=generator.get_state(), torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                    numpy_rng=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                    cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                    seconds=previous_seconds+time.perf_counter()-started,
                    torch_version=str(torch.__version__), device=str(device))

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, request_stop)
        while epoch < config['max_epochs'] and stopper.bad_epochs < stopper.patience:
            model.train()
            total, count = 0., 0
            learning_rate = optimizer.param_groups[0]['lr']
            final_total, auxiliary_total, gradient_norms = 0., 0., []
            term_totals = dict(pcc_loss=0., ssim_loss=0., amplitude_mse=0.)
            for images in train_loader:
                images = images.to(device)
                optimizer.zero_grad(set_to_none=True)
                stages = model(images, return_stages=True)
                final_loss, terms = physical_loss(model.optics(stages[-1]).abs(), images, return_components=True)
                auxiliary = final_loss.new_zeros(())
                if len(stages) > 1:
                    auxiliary = torch.stack([physical_loss(model.optics(field).abs(), images) for field in stages[:-1]]).mean()
                loss = final_loss + config['auxiliary_weight']*auxiliary
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                gradient_norms.append(float(norm))
                optimizer.step()
                global_step += 1
                total += float(loss.detach())*len(images)
                final_total += float(final_loss.detach())*len(images)
                auxiliary_total += float(auxiliary.detach())*len(images)
                for key, value in terms.items():
                    term_totals[key] += float(value.detach())*len(images)
                count += len(images)
            epoch += 1
            measured = validate(model, val_loader, device, precise, config['diagnostic_samples'])
            should_stop = stopper.update(measured['loss'])
            if scheduler is not None:
                scheduler.step(measured['loss'])
            is_best = measured['loss'] < best_loss
            if is_best:
                best_loss = measured['loss']
            is_best_pcc = measured['mean_pcc'] is not None and measured['mean_pcc'] > best_pcc
            is_best_ssim = measured['mean_ssim'] > best_ssim
            if is_best_pcc:
                best_pcc = measured['mean_pcc']
            if is_best_ssim:
                best_ssim = measured['mean_ssim']
            stage_history.extend(dict(epoch=epoch, **row) for row in measured['diagnostics'])
            history.append(dict(epoch=epoch, global_step=global_step, train_loss=total/count,
                                train_final_loss=final_total/count, train_aux_loss=auxiliary_total/count,
                                learning_rate=learning_rate, next_learning_rate=optimizer.param_groups[0]['lr'],
                                gradient_norm_mean=sum(gradient_norms)/len(gradient_norms),
                                gradient_norm_max=max(gradient_norms),
                                gradient_clipped_fraction=sum(v > 5. for v in gradient_norms)/len(gradient_norms),
                                **{f'train_{key}':value/count for key, value in term_totals.items()},
                                **{f'val_{key}':value for key, value in measured['components'].items()},
                                val_loss=measured['loss'], val_pcc=measured['mean_pcc'], val_ssim=measured['mean_ssim'],
                                bad_epochs=stopper.bad_epochs,
                                **{f'val_pcc_{k+1}':v for k,v in enumerate(measured['pcc_planes'])},
                                **{f'val_ssim_{k+1}':v for k,v in enumerate(measured['ssim_planes'])}))
            state = checkpoint()
            if is_best:
                save_checkpoint(root/'best.pt', dict(state, selection_criterion='validation_loss'))
            # Additional best files are compact inference checkpoints. last.pt
            # contains AdamW/scheduler/RNG and is the file used for resuming.
            inference = {key:state[key] for key in ('format', 'config', 'model', 'epoch', 'global_step',
                                                   'dataset_fingerprint', 'metric_selection_start_epoch')}
            inference['validation'] = {key:measured[key] for key in ('loss', 'mean_pcc', 'mean_ssim')}
            if is_best_pcc:
                save_checkpoint(root/'best_pcc.pt', dict(inference, selection_criterion='validation_pcc'))
            if is_best_ssim:
                save_checkpoint(root/'best_ssim.pt', dict(inference, selection_criterion='validation_ssim'))
            # Commit resume state after the corresponding best weights exist.
            save_checkpoint(root/'last.pt', dict(state, selection_criterion='last_epoch'))
            write_json(root/'history.json', history)
            write_csv(root/'history.csv', history)
            write_csv(root/'stage_history.csv', stage_history)
            write_json(root/'diagnostics.json', dict(epoch=epoch, split='val',
                       sample_ids=[example['sample'] for example in measured['examples']],
                       stages=measured['diagnostics']))
            plot_validation(root, epoch, measured['examples'])
            plot_history(root, history)
            status = dict(stage='training', epoch=epoch, global_step=global_step,
                          bad_epochs=stopper.bad_epochs, best_val_loss=best_loss,
                          next_learning_rate=optimizer.param_groups[0]['lr'])
            print(f"Epoch {epoch}/{config['max_epochs']} | train_final={final_total/count:.6f} aux={auxiliary_total/count:.6f} val={measured['loss']:.6f} | PCC={measured['mean_pcc']} SSIM={measured['mean_ssim']:.6f} | lr={learning_rate:.2g}->{optimizer.param_groups[0]['lr']:.2g} grad={sum(gradient_norms)/len(gradient_norms):.3g} | stale={stopper.bad_epochs}/{stopper.patience}", flush=True)
            write_json(root/'status.json', status)
            if backup_dir is not None:
                backup_run(root, backup_dir, epoch)
            if requested_stop:
                write_json(root/'status.json', {**status, 'stage':'interrupted'})
                return root
            if should_stop:
                break
        reason = 'validation_patience' if stopper.bad_epochs >= stopper.patience else 'max_epochs'
        write_json(root/'status.json', dict(stage='complete', reason=reason, epoch=epoch,
                                          global_step=global_step, best_val_loss=best_loss))
        plot_history(root, history)
        if backup_dir is not None:
            backup_run(root, backup_dir, epoch)
        return root
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


