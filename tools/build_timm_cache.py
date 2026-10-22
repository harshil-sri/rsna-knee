#!/usr/bin/env python3
"""Package local timm/HF DINOv2 cache into kaggle-out/dinov2-timm-cache.tar (~100 MB)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_TAR = REPO / "kaggle-out" / "dinov2-timm-cache.tar"

MODEL = "vit_small_patch14_dinov2.lvd142m"

hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
torch_home = Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser()

candidates = list(hf_home.glob("**/*vit_small_patch14_dinov2*")) + list(
    torch_home.glob("**/*vit_small_patch14_dinov2*")
)
if not candidates:
    print("ERROR: no local timm cache for", MODEL, "- run: timm.create_model(..., pretrained=True) once")
    sys.exit(1)

stage = Path("/tmp/opencode/dinov2-stage")
if stage.exists():
    shutil.rmtree(stage)
hf_dst = stage / "hf"
torch_dst = stage / "torch"
hf_dst.mkdir(parents=True)
torch_dst.mkdir(parents=True)

seen = set()
for c in candidates:
    base_dir = c if c.is_dir() else c.parent
    key = str(base_dir)
    if key in seen:
        continue
    seen.add(key)
    files = sorted(base_dir.rglob("*")) if base_dir.is_dir() else [base_dir]
    if hf_home in base_dir.parents:
        rel = base_dir.relative_to(hf_home)
        dst_root = hf_dst / rel
    elif torch_home in base_dir.parents:
        rel = base_dir.relative_to(torch_home)
        dst_root = torch_dst / rel
    else:
        continue
    dst_root.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.is_file():
            (dst_root / f.relative_to(base_dir)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst_root / f.relative_to(base_dir))
    print("staged", base_dir)

OUT_TAR.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["tar", "cf", str(OUT_TAR), "-C", str(stage), "."], check=True)
shutil.rmtree(stage)
print("wrote", OUT_TAR, f"{OUT_TAR.stat().st_size/1e6:.1f} MB")
