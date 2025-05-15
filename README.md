<a href="https://discrete-distribution-networks.github.io/"><img src="https://img.shields.io/static/v1?label=Page&message=github.io&color=blue"></a>
<a href="https://arxiv.org/abs/2401.00036"><img src="https://img.shields.io/badge/arXiv-2401.00036-b31b1b.svg"></a>
<a href="https://openreview.net/forum?id=xNsIfzlefG"><img src="https://img.shields.io/badge/Accepted-ICLR%202025-brightgreen.svg"></a>
<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg"></a>

<!-- <a href="https://huggingface.co/spaces/"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)"></a> -->

# Discrete Distribution Networks
**A novel generative model with simple principles and unique properties**

![](https://discrete-distribution-networks.github.io/img_for_other_repo/ddn-header-en.png)  
*Left: DDN reconstruction process; Right: DDN fitting a 2D distribution*

## ▮ Introduction
We introduce a novel generative model: Discrete Distribution Networks (DDN).

DDN employs a simple yet unique approach to model the target distribution, distinctly different from mainstream generative models:
1. The model generates multiple outputs simultaneously in a single forward pass, rather than just one output.
2. It utilizes these simultaneously generated outputs to fit the target distribution of the training data.
3. These outputs collectively represent a discrete distribution, which is the origin of the name "Discrete Distribution Networks."

Every generative model has its unique characteristics, and DDN is no exception. We will highlight two distinctive capabilities of DDN:
- Zero-Shot Conditional Generation without gradient computation.
- Tree-Structured 1D Discrete Latent variables.

See:
- Paper: https://arxiv.org/abs/2401.00036  
- Page: https://discrete-distribution-networks.github.io/


![](https://discrete-distribution-networks.github.io/img/zscg.png)  
*DDN's feature: Zero-Shot Conditional Generation*

---

The core algorithm implementation of DDN is separately housed in the library [**sddn**](https://github.com/diyer22/sddn) for isolation and reuse. Additionally, **`sddn`** includes simple experiments (2D toy data generation and MNIST example).

To run more complex DDN experiments (CIFAR, FFHQ), we integrated DDN with the [NVlabs/EDM](https://github.com/NVlabs/edm) codebase, resulting in this repository. Therefore, the usage of this repo is almost identical to NVlabs/EDM.


## ▮ Preparing
We provide two environment setup options:
1. pip
2. Docker

### pip
Please install the appropriate [PyTorch](https://pytorch.org/get-started/locally/) version according to your CUDA version first
```bash
git clone https://github.com/DIYer22/discrete_distribution_networks.git
cd discrete_distribution_networks
pip install -r requirements.txt
```

### Docker
First install [Docker](https://docs.docker.com/get-started/) and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
```bash
git clone https://github.com/DIYer22/discrete_distribution_networks.git
cd discrete_distribution_networks
docker build --network=host -t diyer22/ddn .
# Enter the docker environment
docker run -it --gpus all --net=host -v `pwd`:/workspace --user $(id -u):$(id -g) diyer22/ddn bash
```


## ▮ Inference
```bash
# cd discrete_distribution_networks
# Download pre-trained CIFAR weights
wget -O weights/cifar-ddn.pkl http://out.diyer22.com:9000/ddn/weights/v15-00035-cifar10-32x32-cifar_blockn32_outputk64_chain.dropout0.05_fp32_goon.v15.22-shot-087808.pkl
# Backup URL: http://113.44.140.251:9000/ddn/weights/v15-00035-cifar10-32x32-cifar_blockn32_outputk64_chain.dropout0.05_fp32_goon.v15.22-shot-087808.pkl

# Generate images using inference
python generate.py --debug 0 --batch=10 --seeds=0-99 --network weights/cifar-ddn.pkl
# Generating 100 images to "weights/generate"...
# Save vis to: weights/cifar-ddn-vis.png
```
More pre-trained weights download links are available in [weights/README.md](weights/README.md)


## ▮ Train
The dataset preparation process is the same as NVlabs/edm. Please follow [NVlabs/edm#preparing-datasets](https://github.com/NVlabs/edm?tab=readme-ov-file#preparing-datasets) to prepare training datasets and fid-refs

```bash
# train CIFAR10 DDN on 8 x A100(80GB)
torchrun --standalone --nproc_per_node=8 train.py --data datasets/cifar10-32x32.zip --fp16=1 --outdir=training-runs --batch-gpu=256 --batch=2048 --desc=cifar_fp16_blockn32_outputk64_chain.dropout0.05 --chain-dropout 0.05 --max-blockn=32 --max-outputk 64

# evaluation using 2 GPUs, if len(seeds)==50000 will auto calculating FID.
torchrun --standalone --nproc_per_node=2 generate.py --seeds=0-49999 --subdirs --batch 128 --network training-runs/00000-cifar10-32x32-cifar_fp16_blockn32_outputk64_chain.dropout0.05/shot-200000.pkl --fid_ref fid-refs/cifar10-32x32.npz
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
- [ ] Upload weights to HF
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