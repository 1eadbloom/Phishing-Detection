#!/usr/bin/env python3

import sys
import os
import runpy
import torch

print('[OK] PyTorch preloaded: ' + str(torch.__version__))
print('[INFO] CUDA available: ' + str(torch.cuda.is_available()))
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

if len(sys.argv) > 1:
    script_path = sys.argv[1]
    sys.argv = [script_path] + sys.argv[2:]
    try:
        runpy.run_path(script_path, run_name='__main__')
    except SystemExit:
        pass
else:
    print('Usage: python torch_preload.py <script.py>')
    sys.exit(1)
