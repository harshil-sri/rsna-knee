#!/bin/bash
# Watcher v17: Tracks versia-11 execution & auto-submits Daily Submission #5

cd /home/harshil/kaggle/wt
OURT=$(cat ~/.kaggle/access_token)
DAK=$(cat ~/.kaggle/daksh_access_token)

log(){ echo "$1" >> watch-folds-recovery.log; }

for r in 1 2 3 4; do
  line="$(TZ=Asia/Kolkata date '+%H:%M')"

  # Poll versia-11 (daksh)
  s_v11=$(KAGGLE_API_TOKEN=$DAK timeout 50 kaggle kernels status dakxsh23/rsna-knee-versia11 2>&1 | grep -o 'KernelWorkerStatus.[A-Z_]*')
  line="$line versia11=$s_v11"

  if [ "$s_v11" = "KernelWorkerStatus.COMPLETE" ] && [ ! -f .versia11_submitted ]; then
    log "VERSIA11-COMPLETE: dakxsh23/rsna-knee-versia11 is COMPLETE at $line — submitting to competition..."
    touch .versia11_submitted
    python3 - << 'PYEOF' >> watch-folds-recovery.log 2>&1
import os, json
from kaggle.api.kaggle_api_extended import KaggleApi

dak_token = open(os.path.expanduser("~/.kaggle/daksh_access_token")).read().strip()
os.environ["KAGGLE_API_TOKEN"] = dak_token
api = KaggleApi()
api.authenticate()

# Read description if available
desc = "versia-11: optimal synthesis candidate (Daily Submission #5)"
try:
    with open("versia11_desc.txt") as f:
        desc = f.read().strip()
except Exception:
    pass

try:
    r = api.competition_submit_code(
        file_name="submission.csv",
        kernel="dakxsh23/rsna-knee-versia11",
        kernel_version=1,
        competition="rsna-knee-abnormality-detection",
        message=desc
    )
    print("VERSIA11-SUBMIT-OK:", r)
except Exception as e:
    resp = getattr(e, "response", None)
    print("VERSIA11-SUBMIT-FAIL:", resp.text[:300] if resp is not None else str(e)[:300])
PYEOF
  fi

  log "$line"
  [ $r -lt 4 ] && sleep 600
done

log "CHUNK-EXIT $(TZ=Asia/Kolkata date '+%H:%M')"
