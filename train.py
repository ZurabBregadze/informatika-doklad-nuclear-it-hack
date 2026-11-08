"""Open this file in PyCharm and press Run. Edit settings below."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT/'src'))

from doe_p0.training import train

DATA_ROOT = PROJECT/'data'/'doe_targets_500_safe'
OUTPUT = PROJECT/'runs'/'joint_spectral_v2_1'
RESUME = None  # To continue: OUTPUT/'last.pt'

CONFIG = dict(
    architecture='joint_spectral_v2',
    width=32,
    feedback_steps=2,
    batch_size=4,
    learning_rate=1e-4,
    weight_decay=1e-4,
    max_epochs=200,
    patience=10,
    min_delta=1e-5,
    seed=0,
    threads=4,
    device='auto',  # 'auto', 'cpu' or 'cuda'. Mac uses CPU for complex FFT.
    auxiliary_weight=.2,
    lr_scheduler='plateau',
    lr_factor=.5,
    lr_patience=2,  # Reduce after 3 non-improving epochs; early stopping remains 10.
    min_lr=1e-6,
    diagnostic_samples=4,
)

if __name__ == '__main__':
    print(train(DATA_ROOT, OUTPUT, CONFIG, resume=RESUME))
