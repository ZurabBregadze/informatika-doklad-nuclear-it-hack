"""Run the basic numerical GS alone from PyCharm."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT/'src'))

from doe_p0.fresnel_baseline import run

INPUT = [PROJECT/'data/doe_targets_500_safe/test/sample_0451'/f'image_{k}.png' for k in range(1, 5)]
OUTPUT = PROJECT/'results'/'gs_only'
ITERATIONS = 200
SEED = 0

if __name__ == '__main__':
    print(run(INPUT, OUTPUT, ITERATIONS, SEED))
