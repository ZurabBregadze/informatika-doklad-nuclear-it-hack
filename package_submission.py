"""Build a GitHub-ready ZIP with source, Docker, weights and supporting reports."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
ROOT_FILES = ('README.md', 'Dockerfile', '.dockerignore', '.gitignore', 'requirements.txt',
              'pyproject.toml', 'infer.py', 'export_weights.py', 'package_submission.py',
              'train.py', 'compare.py', 'predict.py', 'gs.py')
DOC_FILES = ('TRAINING.md', 'RESULTS.md', 'SUBMISSION.md', 'DEFENSE.md')


def submission_files():
    paths = [ROOT/name for name in ROOT_FILES]
    paths.extend(ROOT/'docs'/name for name in DOC_FILES)
    for directory in ('src', 'tests', 'weights', 'examples', 'reports', '.github/workflows'):
        paths.extend(path for path in (ROOT/directory).rglob('*') if path.is_file()
                     and '__pycache__' not in path.parts and path.name!='.DS_Store'
                     and path.suffix not in ('.pyc', '.pyo', '.tmp'))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    model = ROOT/'weights/model.pt'
    metadata = json.loads((ROOT/'weights/model.json').read_text())
    if hashlib.sha256(model.read_bytes()).hexdigest()!=metadata['sha256']:
        raise ValueError('Weights and metadata do not match. Run export_weights.py first.')
    return sorted(set(paths))


def main():
    output = ROOT/'hackathon_submission.zip'
    paths = submission_files()
    with zipfile.ZipFile(output.with_suffix('.tmp'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    output.with_suffix('.tmp').replace(output)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC check failed.')
    print(f'{output}\nFiles: {len(paths)}; size: {output.stat().st_size/2**20:.2f} MiB')


if __name__=='__main__':
    main()
