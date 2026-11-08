"""Submission inference: four target amplitudes -> one 8-bit phase-only PNG."""
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image
import torch

from .phase import encode_phase
from .weights import load_weights, sha256


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2]/'weights/model.pt'


def read_inputs(paths):
    if len(paths) != 4:
        raise ValueError('Exactly four images are required, ordered by distance.')
    images = []
    for path in paths:
        with Image.open(path) as source:
            if source.size != (128, 128) or getattr(source, 'n_frames', 1) != 1:
                raise ValueError(f'Expected a static 128x128 image: {path}')
            if source.mode not in ('1', 'L', 'P', 'RGB', 'RGBA', 'LA'):
                raise ValueError(f'Expected 8-bit grayscale or RGB input: {path} ({source.mode})')
            if source.mode in ('RGBA', 'LA') or 'transparency' in source.info:
                if source.convert('RGBA').getchannel('A').getextrema()[0] != 255:
                    raise ValueError(f'Input must be opaque; flatten transparency first: {path}')
            # For the training format L this preserves every source pixel.
            values = np.array(source.convert('L'), dtype=np.float32)/255
        images.append(torch.from_numpy(values))
    return torch.stack(images)


def discover_inputs(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f'Input directory does not exist: {directory}')
    layouts = [[directory/f'image_{k}.png' for k in range(1, 5)],
               [directory/f'{k}.png' for k in range(1, 5)]]
    complete = [paths for paths in layouts if all(path.is_file() for path in paths)]
    if len(complete) != 1:
        raise ValueError(f'Expected image_1.png ... image_4.png or 1.png ... 4.png in {directory}; '
                         'for other or ambiguous names use --inputs in physical distance order.')
    return complete[0]


def save_phase(codes, output):
    output = Path(output)
    if codes.dtype != torch.uint8 or tuple(codes.shape) != (128, 128):
        raise ValueError('Phase output must be uint8 with shape 128x128.')
    if output.suffix.lower() != '.png':
        raise ValueError('Output filename must end in .png.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.doe_', suffix='.png', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            Image.fromarray(codes.cpu().numpy()).save(stream, format='PNG')
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        # NamedTemporaryFile starts at 0600. Make the final PNG readable by a
        # host evaluator even when Docker creates it as root on a bind mount.
        temporary.chmod(0o644)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run(tasks, checkpoint=DEFAULT_CHECKPOINT, device='auto', threads=2, report_path=None):
    started = time.perf_counter()
    if type(threads) is not int or threads < 1:
        raise ValueError('--threads must be positive.')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested CUDA device is unavailable. Use --device cpu or a GPU Docker runtime.')
    device = torch.device(device)
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    all_inputs = {Path(path).resolve() for task in tasks for path in task['inputs']}
    outputs = [Path(task['output']).resolve() for task in tasks]
    if len(set(outputs)) != len(outputs) or any(path in all_inputs for path in outputs):
        raise ValueError('Output paths must be distinct and must not overwrite input images.')
    if report_path is not None:
        report_path = Path(report_path).resolve()
        if report_path.suffix.lower() != '.json' or report_path in all_inputs or report_path in outputs:
            raise ValueError('--report must be a separate .json file.')
    checkpoint = Path(checkpoint).resolve()
    digest = sha256(checkpoint)
    metadata_path = checkpoint.with_suffix('.json')
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('sha256') != digest:
            raise ValueError('Weight checksum mismatch. Export checkpoint and metadata together with export_weights.py.')
    load_started = time.perf_counter()
    model, saved = load_weights(checkpoint, device)
    synchronize(device)
    report = dict(architecture=model.architecture, checkpoint_sha256=digest,
                  checkpoint_epoch=saved.get('epoch'), device=str(device),
                  gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                  torch_version=str(torch.__version__), threads=threads,
                  model_load_seconds=time.perf_counter()-load_started,
                  synthesis_timing='CPU tensor -> device transfer -> neural forward -> phase -> CPU tensor; no PNG I/O.',
                  total_timing='main run only; Python/import startup excluded; no hidden warmup.', scenes=[])
    with torch.inference_mode():
        for task in tasks:
            read_started = time.perf_counter()
            targets = read_inputs(task['inputs'])
            read_seconds = time.perf_counter()-read_started
            synchronize(device)
            synthesis_started = time.perf_counter()
            phase = torch.angle(model(targets[None].to(device)))[0].cpu()
            synchronize(device)
            synthesis_seconds = time.perf_counter()-synthesis_started
            save_started = time.perf_counter()
            save_phase(encode_phase(phase), task['output'])
            report['scenes'].append(dict(sample=task['sample'],
                inputs=[str(path) for path in task['inputs']], output=str(task['output']),
                read_seconds=read_seconds, synthesis_seconds=synthesis_seconds,
                quantize_and_save_seconds=time.perf_counter()-save_started))
    report['total_seconds'] = time.perf_counter()-started
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        temporary.replace(report_path)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--input-dir', type=Path, help='One directory with image_1.png ... image_4.png (or 1.png ... 4.png)')
    source.add_argument('--inputs', nargs=4, type=Path, metavar=('Z1', 'Z2', 'Z3', 'Z4'), help='Four explicit paths in 24.1/25.3/26.5/27.7 mm order')
    source.add_argument('--input-root', type=Path, help='Batch root with one subdirectory per four-image set')
    parser.add_argument('--output', type=Path, help='Single PNG output; default /output/doe_phase.png')
    parser.add_argument('--output-dir', type=Path, help='Batch outputs: DIR/<set>/doe_phase.png')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--report', type=Path, help='Optional JSON with per-set timings')
    args = parser.parse_args(argv)
    try:
        if args.input_root is not None:
            if args.output_dir is None or args.output is not None:
                raise ValueError('Batch mode requires --output-dir and does not use --output.')
            folders = sorted(path for path in args.input_root.iterdir() if path.is_dir() and not path.name.startswith('.'))
            if not folders:
                raise ValueError('No image-set subdirectories found.')
            tasks = [dict(sample=path.name, inputs=discover_inputs(path),
                          output=args.output_dir/path.name/'doe_phase.png') for path in folders]
        else:
            if args.output_dir is not None:
                raise ValueError('--output-dir is used only with --input-root.')
            inputs = args.inputs if args.inputs is not None else discover_inputs(args.input_dir or Path('/input'))
            tasks = [dict(sample='sample', inputs=inputs, output=args.output or Path('/output/doe_phase.png'))]
        report = run(tasks, args.checkpoint, args.device, args.threads, args.report)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f'Error: {error}\n')
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__=='__main__':
    main()
