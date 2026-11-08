"""Run after train.py to export neural/GS PNGs and metrics on test only."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT/'src'))

from doe_p0.comparison import compare

DATA_ROOT = PROJECT/'data'/'doe_targets_500_safe'
CHECKPOINT = PROJECT/'runs'/'joint_spectral_v2_1'/'best.pt'
OUTPUT = PROJECT/'results'/'joint_spectral_v2_1'
GS_ITERATIONS = 200
GS_SEED = 0
LIMIT = None  # None = all 50 test scenes. Example: 5 for a preliminary comparison.
DEVICE = 'auto'
TIMING_SAMPLES = 3
TIMING_REPEATS = 3

if __name__ == '__main__':
    print(compare(CHECKPOINT, DATA_ROOT, OUTPUT, GS_ITERATIONS, GS_SEED, LIMIT, DEVICE,
                  timing_samples=TIMING_SAMPLES, timing_repeats=TIMING_REPEATS))
