#!/bin/bash
cd /home/harshil/kaggle/rsna-knee
mkdir -p logs
stamp() { date +%H:%M:%S; }
echo "RUNNING zip-download $(stamp)" > logs/STATUS
python3 -u tools/zip_subset.py --data-dir /mnt/ssd/rsna-knee/data --workers 48 download >> logs/download.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "FAILED zip-download rc=$rc $(stamp)" >> logs/STATUS
  exit $rc
fi
echo "RUNNING verify $(stamp)" >> logs/STATUS
python3 -u tools/download_subset.py --data-dir /mnt/ssd/rsna-knee/data verify >> logs/verify.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "FAILED verify rc=$rc $(stamp)" >> logs/STATUS
  exit $rc
fi
echo "DONE download+verify $(stamp)" >> logs/STATUS
exec bash tools/night_train.sh
