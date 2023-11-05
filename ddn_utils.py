import numpy

try:  # Compatible with s3
    import brainpp_yl.fs

    brainpp_yl.fs.compat_mode()
except:
    pass

import os
import sys
import boxx
import torch

sys.path.append(os.path.abspath("."))
cudan = torch.cuda.device_count()
if cudan <= 1:
    args, argkv = boxx.getArgvDic()
else:
    argkv = {}
debug = (
    not cudan or torch.cuda.get_device_properties("cuda:0").total_memory / 2**30 < 10
)

if argkv.get("debug"):
    debug = True
