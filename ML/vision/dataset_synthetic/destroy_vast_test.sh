#!/bin/bash
# One-shot scheduled destroy for the vast_test instance (162.43.139.39:50803,
# Vast.ai instance ID 50331222), fired by a date-scoped cron entry for
# 2026-09-09 06:00 local time. Not part of the main fleet -- this box only
# ever runs prompt_experiment.py / gemini_image_test.py testing, never the
# production dataset job.
cd /Users/sayyed/development/repos/LookMax/ML/vision/dataset_synthetic
/Users/sayyed/miniconda3/bin/python3 -c "
import sys
sys.path.insert(0, '.')
from fleet_monitor import destroy_vast_instance, get_vast_api_key
key = get_vast_api_key()
ok = destroy_vast_instance(50331222, key)
print('vast_test (50331222) destroyed:', ok)
"
