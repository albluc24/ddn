# 离散分布网络
**全新的生成模型，带来独一无二的能力**

![](https://discrete-distribution-networks.github.io/img/ddn-intro.png)



![](https://discrete-distribution-networks.github.io/img/2d-density.png)

ICLR 2025 论文的官方 PyTorch 实现


## Introduction
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

---
## Getting started with MNIST demo


# Let's train a DDN with 

## Requirements
DDN 的实现是基于 [NVlabs/EDM](https://github.com/NVlabs/edm) codebase 修改而来。配置和 NVlabs/EDM 相同。

目前仅在 Linux + 2080Ti/A800 GPU 上通过测试


## Getting started



