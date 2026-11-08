"""Two alternating checkpoint archives for Colab runtime interruptions."""
import json
from pathlib import Path
import shutil
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, BadZipFile

FILES = ('last.pt', 'best.pt', 'best_pcc.pt', 'best_ssim.pt', 'config.json', 'dataset_audit.json',
         'history.json', 'history.csv', 'status.json', 'learning_curves.png',
         'stage_history.csv', 'diagnostics.json', 'validation_examples.png')


def backup_run(run_dir, backup_dir, epoch):
    run_dir, backup_dir = Path(run_dir), Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    destination = backup_dir / f'checkpoint_{epoch % 2}.zip'
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary)/'checkpoint.zip'
        with ZipFile(archive, 'w', ZIP_DEFLATED) as output:
            for name in FILES:
                if (run_dir/name).is_file():
                    output.write(run_dir/name, name)
            output.writestr('snapshot.json', json.dumps(dict(epoch=epoch)))
        # Keep the other slot intact if Drive disconnects during this copy.
        shutil.copyfile(archive, destination)
    print(f'Backup saved: epoch {epoch} -> {destination.name}', flush=True)


def restore_run(backup_dir, run_dir):
    """Restore the newest valid slot to a fresh local directory."""
    backup_dir, run_dir = Path(backup_dir), Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError('Restore requires an empty local run directory.')
    candidates = []
    for archive in backup_dir.glob('checkpoint_*.zip'):
        try:
            with ZipFile(archive) as saved:
                allowed = set(FILES) | {'snapshot.json'}
                if not set(saved.namelist()).issubset(allowed):
                    raise ValueError(f'Unexpected files in backup: {archive}')
                if not {'last.pt', 'best.pt', 'config.json'}.issubset(saved.namelist()):
                    continue
                if saved.testzip() is not None:
                    continue
                epoch = int(json.loads(saved.read('snapshot.json'))['epoch'])
                candidates.append((epoch, archive))
        except (BadZipFile, OSError, KeyError, ValueError):
            continue
    if not candidates:
        return None
    epoch, archive = max(candidates, key=lambda item: item[0])
    run_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as saved:
        for name in FILES:
            if name in saved.namelist():
                (run_dir/name).write_bytes(saved.read(name))
    print(f'Restored checkpoint: epoch {epoch}', flush=True)
    return run_dir/'last.pt'
