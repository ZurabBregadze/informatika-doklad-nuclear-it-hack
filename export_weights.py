"""Export a trained cluster checkpoint into the self-contained submission."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))

from doe_p0.weights import export_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, help='Trained best.pt or best_ssim.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'weights/model.pt')
    args = parser.parse_args()
    print(json.dumps(export_weights(args.checkpoint, args.output), indent=2))


if __name__=='__main__':
    main()
