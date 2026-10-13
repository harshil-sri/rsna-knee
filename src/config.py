import os
from pathlib import Path

DATA_ROOT = os.environ.get("RSNA_DATA_ROOT", "/mnt/ssd/rsna-knee/data")
IMAGE_SIZE = 224
SLICES_PER_WINDOW = 5
WINDOWS_PER_STUDY_TRAIN = 42
WINDOWS_PER_STUDY_SMOKE = 8
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2
LR = 1e-4
WEIGHT_DECAY = 1e-5
SEED = 42
AMP_DTYPE = "bf16"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "checkpoints"
SMOKE_FAKE_DIR = Path("/tmp/opencode/rsna_fake")
