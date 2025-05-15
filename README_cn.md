<a href="https://discrete-distribution-networks.github.io/"><img src="https://img.shields.io/static/v1?label=Page&message=github.io&color=blue"></a>
<a href="https://arxiv.org/abs/2401.00036"><img src="https://img.shields.io/badge/arXiv-2401.00036-b31b1b.svg"></a>
<a href="https://openreview.net/forum?id=xNsIfzlefG"><img src="https://img.shields.io/badge/Accepted-ICLR%202025-brightgreen.svg"></a>
<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg"></a>

<!-- <a href="https://huggingface.co/spaces/"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)"></a> -->

# 离散分布网络
**全新的生成模型，带来独一无二的能力**

![](https://discrete-distribution-networks.github.io/img_for_other_repo/ddn-header-cn.png)  
*左图：DDN 重建过程；右图：DDN 拟合二维分布*

## ▮ Introduction
我们提出了一种全新的生成模型：离散分布网络（Discrete Distribution Networks），简称 DDN。

DDN 采用一种简洁且独特的方法来建模目标分布，与主流生成模型截然不同：
1. 模型在一次前向过程中同时生成多个输出，而不仅仅是一个输出。
2. 利用这些一次性生成的多个输出来拟合训练数据的目标分布。
3. 这些输出共同表示一个离散分布，这也是“离散分布网络”名称的由来。

每个生成模型都有其独特的特性，DDN 也不例外。我们将重点介绍 DDN 的两个特有能力：
- 无需计算梯度即可实现 零样本条件生成（Zero-Shot Conditional Generation）。
- 具有树状结构的一维离散潜变量（Tree-Structured 1D Discrete Latent）。

参见：
- Paper: https://arxiv.org/abs/2401.00036  
- Page: https://discrete-distribution-networks.github.io/


![](https://discrete-distribution-networks.github.io/img/zscg.png)  
*DDN 的特色：零样本条件生成*

---

DDN 核心算法的代码实现单独放在了库 [**sddn**](https://github.com/sddn) 中，以方便隔离和复用。此外，**`sddn`** 也包括了简单的实验 (2D toy data generation and MNIST example).

为了跑更复杂的 DDN 实验 (CIFAR、FFHQ)，我们在 [NVlabs/EDM](https://github.com/NVlabs/edm) 的 codebase 上整合了 DDN，从而诞生本 repo。所以本 repo 的用法和 NVlabs/EDM 几乎一致。


## ▮ Preparing
我们提供两种环境安装方案
1. pip
2. Docker

### pip
请先根据你的 CUDA 版本安装对应的 [PyTorch](https://pytorch.org/get-started/locally/)
```bash
git clone https://github.com/DIYer22/discrete_distribution_networks.git
cd discrete_distribution_networks
pip install -r requirements.txt
```

### Docker
首先安装好 [Docker](https://docs.docker.com/get-started/) 和 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
```bash
git clone https://github.com/DIYer22/discrete_distribution_networks.git
cd discrete_distribution_networks
docker build --network=host -t diyer22/ddn .
# 进入 docker 环境
docker run -it --gpus all --net=host -v `pwd`:/workspace --user $(id -u):$(id -g) diyer22/ddn bash
```


## ▮ Inference
```bash
# cd discrete_distribution_networks
# 下载训练好的 CIFAR 权重
wget -O weights/cifar-ddn.pkl http://113.44.140.251:9000/ddn/weights/v15-00035-cifar10-32x32-cifar_blockn32_outputk64_chain.dropout0.05_fp32_goon.v15.22-shot-087808.pkl

# Inference 生成图片
python generate.py --debug 0 --batch=10 --seeds=0-99 --network weights/cifar-ddn.pkl
# Generating 100 images to "weights/generate"...
# Save vis to: weights/cifar-ddn-vis.png
```
更多的权重下载地址在 [weights/README.md](weights/README.md)


## ▮ Train
数据集准备流程和 NVlabs/edm 一样, 请根据 [NVlabs/edm#preparing-datasets](https://github.com/NVlabs/edm?tab=readme-ov-file#preparing-datasets) 来准备 training datasets 和 fid-refs

```bash
# train CIFAR10 DDN on 8 x A100(80GB)
torchrun --standalone --nproc_per_node=8 train.py --data datasets/cifar10-32x32.zip --fp16=1 --outdir=training-runs --batch-gpu=256 --batch=2048 --desc=cifar_fp16_blockn32_outputk64_chain.dropout0.05 --chain-dropout 0.05 --max-blockn=32 --max-outputk 64

# evaluation, if len(seeds)==50000 will auto calculating FID.
torchrun --standalone --nproc_per_node=8 generate.py --seeds=0-49999 --subdirs --batch 128 --network training-runs/00000-cifar10-32x32-cifar_fp16_blockn32_outputk64_chain.dropout0.05/shot-200000.pkl --fid_ref fid-refs/cifar10-32x32.npz
# Calculating FID...
# 51.856
# Saving example images tar to: xxx/sample-example.tar
# Save vis to: xxx/vis.png
```

## Misc
### TODO
- [x] Dockerfile
- [x] Inference
- [x] Train
- [ ] MNIST and 2d toy data
- [ ] Zero-Shot Conditional Generation
- [ ] Support CPU inference
- [ ] Online demo of ZSCG on coloring task
### Citation
```bibtex
@inproceedings{yang2025discrete,
  title     = {Discrete Distribution Networks},
  author    = {Lei Yang},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025},
  url       = {https://openreview.net/forum?id=xNsIfzlefG}}
```

