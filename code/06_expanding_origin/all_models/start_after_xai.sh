#!/bin/sh
# Waits until the E7 sequence-model XAI (35 runs) has finished on the GPU, then starts run_windows.py.
XLOG=/c/Users/User/Desktop/Submission_version/Figures_Tables_Fixed/XAI/xai_E7_seq_log.txt
cd /c/Users/User/Desktop/Submission_version/Expanding_origin_all_models
while [ "$(grep -cE 'E7 +[a-z_]+ +seed [0-9]+ ' "$XLOG")" -lt 35 ]; do sleep 60; done
echo "XAI finished at $(date '+%Y-%m-%d %H:%M'); starting run_windows.py" >> run_log.txt
PYTHONIOENCODING=utf-8 /c/Users/User/.conda/envs/duplicates/python.exe -u run_windows.py > run_stdout.txt 2>&1
echo "run_windows.py finished at $(date '+%Y-%m-%d %H:%M')" >> run_log.txt
