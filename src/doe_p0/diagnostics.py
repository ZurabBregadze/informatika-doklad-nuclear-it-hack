"""Small validation previews and atomic training tables; no test-set access."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write_csv(path, rows):
    """Union of columns also supports resuming a run with older history rows."""
    path = Path(path)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def stage_row(sample, stage, result):
    return dict(sample=sample, stage=stage, mean_pcc=result['mean_pcc'], mean_ssim=result['mean_ssim'],
                **{f'pcc_{p["plane"]}':p['pcc'] for p in result['planes']},
                **{f'ssim_{p["plane"]}':p['ssim'] for p in result['planes']})


def plot_validation(root, epoch, examples):
    if not examples:
        return
    columns = 1+len(examples[0]['reconstructions'])
    fig, axes = plt.subplots(4*len(examples), columns,
                             figsize=(2.3*columns, 7*len(examples)), squeeze=False,
                             constrained_layout=True)
    for sample_index, example in enumerate(examples):
        for plane in range(4):
            row = axes[4*sample_index+plane]
            target = example['targets'][plane]
            row[0].imshow((target/target.max()).numpy(), cmap='gray', vmin=0, vmax=1)
            row[0].set_title(f'{example["sample"]}\nTarget {plane+1}', fontsize=8)
            for stage, (codes, result) in enumerate(zip(example['reconstructions'], example['metrics'])):
                metric = result['planes'][plane]
                pcc = 'undefined' if metric['pcc'] is None else f'{metric["pcc"]:.3f}'
                row[stage+1].imshow(codes[plane].numpy()/255, cmap='gray', vmin=0, vmax=1)
                row[stage+1].set_title(f'Stage {stage}: PCC={pcc}\nSSIM={metric["ssim"]:.3f}', fontsize=8)
            for axis in row:
                axis.axis('off')
    fig.suptitle(f'Fixed validation scenes, epoch {epoch}; metrics after 8-bit quantization')
    fig.savefig(Path(root)/'validation_examples.png', dpi=100)
    plt.close(fig)


def plot_history(root, history):
    if not history:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    epochs = [row['epoch'] for row in history]

    def values(key):
        return [float('nan') if row.get(key) is None else row[key] for row in history]

    for key, label in (('train_final_loss', 'Train final'), ('train_loss', 'Train + auxiliary'),
                       ('val_loss', 'Validation final')):
        axes[0, 0].plot(epochs, values(key), '.-', label=label)
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    for axis, key, label in ((axes[0, 1], 'val_pcc', 'Validation PCC'),
                              (axes[0, 2], 'val_ssim', 'Validation SSIM')):
        axis.plot(epochs, values(key), '.-')
        axis.set_ylabel(label)
    for key, label, weight in (('val_pcc_loss', '1-PCC', 1.), ('val_ssim_loss', '1-SSIM', 1.),
                               ('val_amplitude_mse', '0.1 amplitude MSE', .1)):
        axes[1, 0].plot(epochs, [weight*v for v in values(key)], '.-', label=label)
    axes[1, 0].set_ylabel('Validation loss components')
    axes[1, 0].legend()
    axes[1, 1].plot(epochs, values('learning_rate'), '.-')
    axes[1, 1].set(yscale='log', ylabel='Learning rate used')
    axes[1, 2].plot(epochs, values('gradient_norm_mean'), '.-', label='Mean')
    axes[1, 2].plot(epochs, values('gradient_norm_max'), '.-', label='Maximum')
    axes[1, 2].axhline(5., color='gray', linestyle='--', label='Clip threshold')
    axes[1, 2].set_ylabel('Gradient norm before clipping')
    axes[1, 2].legend()
    for axis in axes.flat:
        axis.set_xlabel('Epoch')
    fig.savefig(Path(root)/'learning_curves.png', dpi=130)
    plt.close(fig)
