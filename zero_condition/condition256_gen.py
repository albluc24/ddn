#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 12:56:20 2025

@author: DIYer22
"""

import os
import boxx
from boxx import *
sys.path.append('..')
with inpkg():
    from .main import *
    from ..ddn_utils import *
    from ..training.dataset import ImageFolderDataset

# pklp = "../../asset/v15-00018-ffhq-64x64-blockn64_outputk512_chain.dropout0.05-shot-117913.pkl"
d_init = dict(batch_size=2)
pklp = "../../asset/v16-00009-ffhq-256x256-ffhq256_cond.edge_chain.dropout0.05-shot-055193.pkl"
pklp = "../../asset/v32-init-00001-ffhq-256x256-ffhq256_cond.color_chain.dropout0.05_batch128-shot-025089.pkl"
# pklp = "../../asset/v32-init-00001-ffhq-256x256-ffhq256_cond.color_chain.dropout0.05_batch128-shot-045159.pkl"
pklp = "../../asset/v32-init-00003-ffhq-256x256-ffhq256_cond.color_chain.dropout0.05_batch64_k64-shot-025088.pkl"
pklp = "../../asset/v32-init-00001-ffhq-256x256-ffhq256_cond.color_chain.dropout0.05_batch128-shot-087809.pkl"
pklp = "../../asset/v32-00003-ffhq-256x256-ffhq256_cond.color_chain.dropout0.05_batch64_k64-shot-070246.pkl"
pklp = "../../asset/v32-00003-ffhq-256x256-ffhq256_cond.color_chain.dropout0.05_batch64_k64-shot-122931.pkl"
# pklp = "../../asset/v32-00004-ffhq-256x256-ffhq256_cond.edge_chain.dropout0.05_batch64_k64-shot-047667.pkl"
# pklp = "../../asset/v32-00004-ffhq-256x256-ffhq256_cond.edge_chain.dropout0.05_batch64_k64-shot-092826.pkl"


from torchvision.transforms.functional import InterpolationMode
                                     
img_dir = '/home/yl/dataset/ffhq/test_self/test_self'
# img_dir = '/home/yl/dataset/ffhq/ffhq_small_test'
img_dir = '/home/yl/dataset/ffhq/ffhq_small_test2'
# img_dir = '/home/yl/dataset/ffhq/celeba_small_test'
# img_dir = '/home/yl/dataset/ishape/ishape_dataset/wire/val/image'
# img_dir = '/tmp/a'
dataset = ImageFolderDataset(img_dir)
samples_per_condition = 3
# slicee = slice(0,4,)
slicee = slice(-4, None)
condition_source = tht([dataset[i][0] for i in range(len(dataset))[slicee]]*samples_per_condition).to(torch.float32) / 127.5 - 1

# resize
# condition_source = resize(condition_source, 256, 'bicubic')
# condition_source = resize(condition_source, 256, 'area')
width, height = 256,256
import PIL.Image   
condition_source = torch.cat([uint8_to_tensor(np.array(PIL.Image.fromarray(arr, "RGB").resize((width, height), PIL.Image.Resampling.LANCZOS)))[None] for arr in t2rgb(condition_source)])


d_init = dict(condition_source=condition_source)
# d_init['target'] = condition_source
# d_init['target'] = condition_source*0 + -tht([-1,-1,-1])[...,None,None]
# d_init['target'] = condition_source[list(range(1, len(condition_source)))+[0]]

net = sys._getframe(6).f_globals.get("net")  # for ipython, to avoid loading weight again
if net is None:
    print('loading weight')
    net = load_net(pklp)
    print(net.model.table())

d = net(d_init)
del d['outputs']

# tree-d

# shows(d['condition_source'], d['predict'], d['condition'], t2rgb, png=True)
shows(d['condition'][:len(d['condition'])//samples_per_condition ], d['predict'], t2rgb, png=True)

if __name__ == "__main__":
    pass
