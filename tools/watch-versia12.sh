#!/bin/bash
# Watcher v19: Tracks versia-12 execution & auto-submits Daily Submission #1 for Wednesday Oct 21 UTC

cd /home/harshil/kaggle/wt
OURT=$(cat ~/.kaggle/access_token)
DAK=$(cat ~/.kaggle/daksh_access_token)

log(){ echo "$1" >> watch-folds-recovery.log; }

for r in 1 2 3 4; do
  line="$(TZ=Asia/Kolkata date '+%H:%M')"

  # Poll versia-12 (daksh)
  s_v12=$(KAGGLE_API_TOKEN=$DAK timeout 50 kaggle kernels status dakxsh23/rsna-knee-versia12 2>&1 | grep -o 'KernelWorkerStatus.[A-Z_]*')
  line="$line versia12=$s_v12"

  if [ "$s_v12" = "KernelWorkerStatus.COMPLETE" ] && [ ! -f .versia12_submitted ]; then
    log "VERSIA12-COMPLETE: dakxsh23/rsna-knee-versia12 is COMPLETE at $line — submitting to competition..."
    touch .versia12_submitted
    python3 - << 'PYEOF' >> watch-folds-recovery.log 2>&1
import os, json
from kaggle.api.kaggle_api_extended import KaggleApi

dak_token = open(os.path.expanduser("~/.kaggle/daksh_access_token")).read().strip()
os.environ["KAGGLE_API_TOKEN"] = dak_token
api = KaggleApi()
api.authenticate()

desc = "versia-12: Raptor 384+224crop multi-res + R7 ConvNeXt flip TTA (0.958+ recipe)"

try:
    r = api.competition_submit_code(
        file_name="submission.csv",
        kernel="dakxsh23/rsna-knee-versia12",
        kernel_version=1,
        competition="rsna-knee-abnormality-detection",
        message=desc
    )
    print("VERSIA12-SUBMIT-OK:", r)
except Exception as e:
    resp = getattr(e, "response", None)
    print("VERSIA12-SUBMIT-FAIL:", resp.text[:300] if resp is not None else str(e)[:300])
PYEOF
  fi

  log "$line"
  [ $r -lt 4 ] && sleep 600
done

log "CHUNK-EXIT $(TZ=Asia/Kolkata date '+%H:%M')"
