"""Compare bundled weights with GS on the prepared test split."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT/'src'))

from doe_p0.comparison import compare

DATA_ROOT = PROJECT/'data'/'main_subset'
CHECKPOINT = PROJECT/'weights'/'model.pt'
OUTPUT = PROJECT/'results'/'current_model_gs200'
GS_ITERATIONS = 200
GS_SEED = 0
LIMIT = None  # All scenes in the supplied test split.
DEVICE = 'auto'
TIMING_SAMPLES = 3
TIMING_REPEATS = 3

if __name__ == '__main__':
    print(compare(CHECKPOINT, DATA_ROOT, OUTPUT, GS_ITERATIONS, GS_SEED, LIMIT, DEVICE,
                  timing_samples=TIMING_SAMPLES, timing_repeats=TIMING_REPEATS))
