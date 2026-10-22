import subprocess
import sys
from pathlib import Path
import pandas as pd

def run_cmd(cmd):
    print(f"--> Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, check=True)
    return res

def main():
    root = Path(__file__).resolve().parent.parent
    fake_dir = Path("/tmp/opencode/rsna_fake")

    run_cmd([sys.executable, str(root / "tools" / "make_fake_data.py")])
    run_cmd([sys.executable, "-m", "src.train", "--smoke"])

    best_pt = root / "checkpoints" / "best.pt"
    assert best_pt.exists(), f"{best_pt} does not exist!"

    preds_csv = fake_dir / "preds.csv"
    run_cmd([
        sys.executable,
        "-m",
        "src.infer",
        "--smoke",
        "--ckpt",
        str(best_pt),
        "--out",
        str(preds_csv),
        "--tta-flip",
    ])

    assert preds_csv.exists(), f"{preds_csv} does not exist!"
    df = pd.read_csv(preds_csv)
    print("preds.csv shape:", df.shape)
    print("preds.csv head:")
    print(df.head())

if __name__ == "__main__":
    main()
