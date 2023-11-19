#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 00:36:09 2023

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
    condidat_num = 100
    target_shape = 64, 64
    if debug:
        condidat_num = 1

    net = sys._getframe(3).f_globals.get("net")
    if net is None:
        print("load net....")
        net = load_net(pklp)
