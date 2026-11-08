"""Compare exported neural and GS masks on the held-out test split."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .data import Scenes, audit_dataset
from .fresnel_baseline import synthesize
from .metrics import export_scene, write_json
from .training import load_model, seed_everything, select_device
from .timing import benchmark


def comparison_png(folder, neural, gs):
    fig, axes = plt.subplots(4, 5, figsize=(13, 10), constrained_layout=True)
    for k in range(1, 5):
        paths = [folder/'neural'/f'target_{k}.png', folder/'neural'/f'reconstruction_{k}.png',
                 folder/'gs'/f'reconstruction_{k}.png', folder/'neural'/f'error_{k}.png', folder/'gs'/f'error_{k}.png']
        titles = [f'Target {k}', 'Neural', 'GS', '|Target - Neural|', '|Target - GS|']
        for col, (path, title) in enumerate(zip(paths, titles)):
            with Image.open(path) as image:
                values = np.array(image, dtype=float)/255
            if col == 0:
                values /= max(values.max(), 1e-12)
            axes[k-1, col].imshow(values, cmap='gray', vmin=0, vmax=1)
            axes[k-1, col].axis('off')
            if col in (1, 2):
                row = (neural if col == 1 else gs)['planes'][k-1]
                pcc = 'undefined' if row['pcc'] is None else f"{row['pcc']:.4f}"
                title += f"\nPCC={pcc} SSIM={row['ssim']:.4f}"
            axes[k-1, col].set_title(title, fontsize=9)
    fig.savefig(folder/'comparison.png', dpi=130)
    plt.close(fig)


def compare(checkpoint, data_root, output, gs_iterations=200, gs_seed=0, limit=None, device='auto', *,
            timing_samples=3, timing_repeats=3, timing_warmup=2):
    if gs_iterations < 1 or (limit is not None and limit < 1):
        raise ValueError('Positive GS iterations and limit required.')
    if (any(type(value) is not int for value in (timing_samples, timing_repeats, timing_warmup))
            or timing_samples < 0 or timing_repeats < 1 or timing_warmup < 1):
        raise ValueError('Check timing samples (0 disables timing), repeats and warmup.')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = select_device(device)
        seed_everything(0)
        torch.set_num_threads(4)
        model, saved = load_model(checkpoint, device)
        audit = audit_dataset(data_root)
        if audit['fingerprint'] != saved['dataset_fingerprint']:
            raise ValueError('Dataset differs from training. Use the same untouched test split.')
        data = Scenes(data_root, 'test')
        count = len(data) if limit is None else min(limit, len(data))
        protocol = dict(format=2, checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                        checkpoint_epoch=saved.get('epoch'), checkpoint_selection=saved.get('selection_criterion', 'legacy'),
                        dataset_fingerprint=audit['fingerprint'], gs_iterations=gs_iterations, gs_seed=gs_seed,
                        sample_ids=[p.name for p in data.folders[:count]], neural_device=str(device),
                        gs_device='cpu', gs_dtype='float64', gs_record_history=False,
                        architecture=model.architecture, torch_version=str(torch.__version__),
                        timing=dict(samples=min(timing_samples, count), repeats=timing_repeats, warmup=timing_warmup),
                        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                        normalization='both images divided by their own maximum', metrics='global PCC/SSIM after PNG roundtrip')
        if (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_text()) != protocol:
            raise ValueError('Comparison protocol changed. Choose a new OUTPUT.')
        write_json(root/'protocol.json', protocol)
        rows = []
        for index in range(count):
            sample_id = data.folders[index].name
            folder = root/sample_id
            if folder.exists():
                if not (folder/'metrics.json').exists():
                    raise ValueError(f'Incomplete sample: {folder}. Choose a new OUTPUT.')
                rows.extend(json.loads((folder/'metrics.json').read_text()))
                continue
            targets = data[index]
            with torch.no_grad():
                phase = torch.angle(model(targets[None].to(device)))[0].cpu()
            gs_phase, _, _, _ = synthesize(targets.double(), iterations=gs_iterations, seed=gs_seed, record_history=False)
            temporary = Path(tempfile.mkdtemp(prefix='scene_', dir=root))
            try:
                neural = export_scene(phase, targets, temporary/'neural')
                gs = export_scene(gs_phase, targets, temporary/'gs')
                comparison_png(temporary, neural, gs)
                sample_rows = []
                for method, result in (('neural', neural), ('gs', gs)):
                    for plane in result['planes']:
                        sample_rows.append(dict(sample=sample_id, method=method, **plane))
                write_json(temporary/'metrics.json', sample_rows)
                temporary.rename(folder)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            rows.extend(sample_rows)
            print(f'{index+1}/{count} {sample_id} | Neural PCC={neural["mean_pcc"]} SSIM={neural["mean_ssim"]:.5f} | GS PCC={gs["mean_pcc"]} SSIM={gs["mean_ssim"]:.5f}', flush=True)
        with (root/'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {}
        for method in ('neural', 'gs'):
            chosen = [row for row in rows if row['method'] == method]
            values = [row['pcc'] for row in chosen]
            summary[method] = dict(samples=count, mean_pcc=None if None in values else float(np.mean(values)),
                                   mean_ssim=float(np.mean([row['ssim'] for row in chosen])),
                                   undefined_pcc=sum(value is None for value in values),
                                   pcc_planes=[None if any(row['pcc'] is None for row in chosen if row['plane']==k)
                                               else float(np.mean([row['pcc'] for row in chosen if row['plane']==k])) for k in range(1, 5)],
                                   ssim_planes=[float(np.mean([row['ssim'] for row in chosen if row['plane']==k])) for k in range(1, 5)])
        write_json(root/'summary.json', summary)
        plot_metrics(root, rows)
        if timing_samples and not (root/'timings.json').exists():
            print('Benchmarking synthesis and in-memory reconstruction pipeline...', flush=True)
            timing = benchmark(model, data, gs_iterations, gs_seed, samples=min(timing_samples, count),
                               repeats=timing_repeats, warmup=timing_warmup)
            timing['checkpoint_sha256'] = protocol['checkpoint_sha256']
            write_json(root/'timings.json', timing)
        write_json(root/'completed.json', dict(completed=True, samples=count))
        return summary


def plot_metrics(root, rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ids = sorted({row['sample'] for row in rows})
    for axis, key in zip(axes, ('pcc', 'ssim')):
        for method in ('neural', 'gs'):
            values = []
            for sample in ids:
                metrics = [row[key] for row in rows if row['sample']==sample and row['method']==method]
                values.append(np.nan if None in metrics else np.mean(metrics))
            axis.plot(range(1, len(ids)+1), values, '.-', label=method)
        axis.set(xlabel='Test scene in CSV order', ylabel=key.upper())
        axis.legend()
    fig.savefig(root/'metrics_comparison.png', dpi=140)
    plt.close(fig)
