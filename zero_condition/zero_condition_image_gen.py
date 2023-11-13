#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:37:03 2023

@author: yanglei
"""

import os
import boxx
from boxx import *

with inpkg():
    from .main import *

if __name__ == "__main__":
    img_glob = "~/dataset/celebA-HQ/test-256x256/*g"
    pklp = "../../asset/v15-00018-ffhq-64x64-blockn64_outputk512_chain.dropout0.05-shot-117913.pkl"
    outputdir = boxx.relfile("../../asset/zero_condition_image_gen/")
    condidat_num = 1
    target_shape = 64, 64

    net = sys._getframe(3).f_globals.get("net")
    if net is None:
        print("load net....")
        net = load_net(pklp)

    imgps = sorted(glob(os.path.expanduser(img_glob)))
    os.makedirs(outputdir, exist_ok=True)
    for imgp in imgps:
        fname = filename(imgp)
        img = boxx.imread(imgp)
        target = (img / 255) * 2 - 1
        target = target.transpose(2, 0, 1)
        target = torch.from_numpy(target).cuda().float()
        target = nn.functional.interpolate(target[None], target_shape, mode="area")[0]
        samplers = {
            "01recon": lambda: L2Sampler(target),
            "02sr.f2": lambda: SuperResSampler(target, 1 / 2),
            "03sr.f4": lambda: SuperResSampler(target, 1 / 4),
            "04sr.f8": lambda: SuperResSampler(target, 1 / 8),
            "05sr.f16": lambda: SuperResSampler(target, 1 / 16),
            "06denoise": lambda: NoiseSampler(target),
            "07lowbit": lambda: LowBitSampler(target),
            "08color": lambda: ColorfulSampler(target),
            "09inpainting_right": lambda: L2MaskedSampler(target, 0),
            "10inpainting_down": lambda: L2MaskedSampler(target, 1),
            "11inpainting_s": lambda: L2MaskedSampler(target, 8),
            "12inpainting_is": lambda: L2MaskedSampler(target, 7),
            "13style": lambda: StyleTransfer(
                target,
            ),
        }
        for sampler_name in samplers:
            print(sampler_name)
            sampler = samplers[sampler_name]()
            batch_sampler = BatchedGuidedSampler(sampler)
            for condidat_idx in range(condidat_num):
                d = dict(sampler=batch_sampler)
                d = net(d)
                showd(d, 1, figsize=(8, 5) if target.shape[-1] >= 40 else (4, 3))
                res = t2rgb(npa(d["predict"][0]))
                target_path = pathjoin(
                    outputdir, f"{fname}_{sampler_name}_{condidat_idx:03}.png"
                )
                boxx.imsave(target_path, res)
            # if "condition" in d
        # show(target, t2rgb)
        # show(img)
        print("All save to")
        break
