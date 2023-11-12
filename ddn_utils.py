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


from boxx import tprgb, show, np, npa

t2rgb = lambda x: (tprgb(npa(x)) * 127.5 + 128).clip(0, 255).astype(np.uint8)
showd = lambda d_, no_predicts=False, **kv: show(
    None if no_predicts else d_.get("predicts", [])[2:],
    d_.get("target"),
    d_["predict"],
    d_.get("condition"),
    d_.get("condition_source"),
    t2rgb,
    **kv,
)

import pickle


def load_net(path, device="cuda"):
    path = path.replace("https://oss.iap.hh-d.brainpp.cn", "s3:/").replace(
        "http://localhost:58000/ddm_exps/", "exps/"
    )
    # pickle.load(open(path, "rb"))["ema"].to("cuda")
    with open(path, "rb") as f:
        if path.endswith(".pkl"):
            net = pickle.load(f)["ema"].to(device)
        elif path.endswith(".pt"):
            net = torch.load(f)["net"].to(device)  # 会保存模型代码吗?
        net = net.eval()
    return net


true, false = True, False
