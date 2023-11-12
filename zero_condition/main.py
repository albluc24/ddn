#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 23:43:43 2023

@author: yanglei
"""
import cv2
import boxx
from boxx.ylth import *
import torch
import random
import torch.nn.functional as F

with boxx.inpkg(), boxx.impt(".."):
    from ddn_utils import *
    import torch_utils, dnnlib


class L2Sampler:
    def __init__(self, target):
        self.raw = target
        self.target = target[None]

    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        resized = nn.functional.interpolate(self.target, (h, w), mode="area")
        probs = nn.functional.softmax(-((rgbs - resized) ** 2).mean([-1, -2, -3]), 0)
        return dict(
            probs=probs,
            idx_k=int(probs.argmax()),
            condition=self.target,
            condition_source=self.raw,
        )


class L1Sampler(L2Sampler):
    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        resized = nn.functional.interpolate(self.target, (h, w), mode="area")
        probs = nn.functional.softmax(
            -(torch.abs(rgbs - resized)).mean([-1, -2, -3]), 0
        )
        return dict(
            probs=probs,
            idx_k=int(probs.argmax()),
            condition=self.target,
            condition_source=self.raw,
        )


def get_mask_s(shape=None):
    canvas = np.zeros((512, 512), dtype=np.uint8)

    # 定义字体类型和大小
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 15

    # 使用cv2.getTextSize获取文字宽度和高度
    thickness = 50
    text_size = cv2.getTextSize("S", font, font_scale, thickness=thickness)[0]

    # 计算文本的基线开始位置，以便让"S"出现在中心
    text_x = (canvas.shape[1] - text_size[0]) // 2 + thickness // 2
    text_y = (canvas.shape[0] + text_size[1]) // 2 - thickness // 5

    # 在画布上绘制文本
    canvas = cv2.putText(
        canvas, "S", (text_x, text_y), font, font_scale, (255), thickness, cv2.LINE_AA
    )
    if shape:
        canvas = cv2.resize(canvas, shape[::-1])
    mask_s = torch.from_numpy(canvas > 128)
    return mask_s


class L2MaskedSampler:
    def __init__(self, target, mask=0):
        self.raw = target
        h, w = target.shape[-2:]
        if isinstance(mask, int):
            mask_ = torch.zeros((h, w)) > 0
            if mask == 0:
                mask_[:, : w // 2] = 1
            if mask == 1:
                mask_[: h // 2] = 1
            if mask == 2:
                mask_[h // 2 :] = 1
            if mask == 3:
                mask_[: h // 2, : w // 2] = 1
                mask_[h // 2 :, w // 2 :] = 1
            if mask == 4:
                mask_[h * 2 // 5 : h * 5 // 9] = 1
            if mask == 5:
                mask_[h * 2 // 5 : h * 5 // 9] = 1
                mask_[:, w * 2 // 5 : w * 3 // 5] = 1
            if mask == 6:
                mask_[
                    h // 4 : h * 3 // 4,
                    w // 4 : w * 3 // 4,
                ] = 1
            if mask == 7:
                mask_ = get_mask_s((h, w))
            if mask == 8:
                mask_ = ~get_mask_s((h, w))
            mask = mask_
        self.mask = mask.cuda()
        self.target = target[None] * self.mask

    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        resized = nn.functional.interpolate(self.target, (h, w), mode="area")
        mask_resized = nn.functional.interpolate(
            self.mask[None, None].float(), (h, w), mode="area"
        )  # [0,0]
        loss = torch.abs(rgbs - resized)  # l1
        # loss = (rgbs-resized)**2 # l2
        probs = nn.functional.softmax(
            -((loss) * mask_resized).sum([-1, -2]).mean(-1)
            / (mask_resized.sum() + eps),
            0,
        )
        return dict(
            probs=probs,
            idx_k=int(probs.argmax()),
            condition=self.target,
            condition_source=self.raw,
        )


class NoiseSampler(L2Sampler):
    def __init__(self, target, noise_rate=0.53):
        self.raw = target
        self.target = target[None] + torch.randn_like(target)[None] * noise_rate


class LowBitSampler(L2Sampler):
    def __init__(self, target, brightnessn=4):
        self.raw = target
        self.target = ((target[None]) * brightnessn / 2).round() * 2 / brightnessn


class ColorfulSampler:
    def __init__(self, target):
        self.raw = target
        self.target = target[None].mean(-3, keepdims=True)

    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        resized = nn.functional.interpolate(self.target, (h, w), mode="area")
        rgbs = rgbs.mean(-3, keepdims=True)
        probs = nn.functional.softmax(-((rgbs - resized) ** 2).mean([-1, -2, -3]), 0)
        return dict(
            probs=probs,
            idx_k=int(probs.argmax()),
            condition=self.target,
            condition_source=self.raw,
        )


class SuperResSampler:
    def __init__(self, target, shape=0.5):
        self.raw = target
        self.target = boxx.resize(target[None], shape, "area")

    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        hh, ww = self.target.shape[-2:]
        resized = nn.functional.interpolate(rgbs, (hh, ww), mode="area")
        probs = nn.functional.softmax(
            -((self.target - resized) ** 2).mean([-1, -2, -3]), 0
        )
        return dict(
            probs=probs,
            idx_k=int(probs.argmax()),
            condition=self.target,
            condition_source=self.raw,
        )


class CannySampler:
    def __init__(self, target):
        self.raw = target
        self.f = lambda timg: cv2.Canny(t2rgb(timg), 100, 200)
        self.target = self.f(target)
        self.cache = {self.target.shape: self.target}

    def __call__(self, dic):
        rgbs = dic["rgbs"]

        k, h, w, c = rgbs.shape
        hh, ww = self.target.shape[-2:]
        rgbs = nn.functional.interpolate(rgbs, (hh, ww), mode="nearest")
        rgbs = (
            rgbs.permute(0, 2, 3, 1).cpu().numpy()
        )  # Change tensor shape to (k, h, w, c)

        # if (h, w) not in self.cache:
        #     raw_resized = nn.functional.interpolate(
        #                self.raw[None], (h, w), mode="area"
        #             )[0]
        #     self.cache[(h, w)] = self.f(raw_resized)
        # target_resized=self.cache[(h, w)]
        target_resized = self.target

        resized_imgs = []
        for img in rgbs:
            resized_img = self.f(img)
            resized_imgs.append(resized_img)

        resized_imgs = np.stack(
            resized_imgs, axis=0
        )  # Stacking the images into one numpy array
        # if w == 64:
        #     g()/0
        if not increase("edge show") % 8:
            show(resized_imgs[:1], rgbs[:1], t2rgb)
        # probs = -(np.abs(resized_imgs - target_resized).clip(0.)).mean(-1).mean(-1)
        # probs = -(np.abs(resized_imgs[target_resized>0]).clip(0.)).mean(-1)#.mean(-1)
        probs = resized_imgs[:, target_resized > 0].mean(-1) + (
            1 - resized_imgs[:, target_resized < 0.5].mean(-1)
        )
        # probs = - probs
        if h <= 16:
            probs += resized_imgs.mean(-1).mean(-1)
        g()
        # probs = F.softmax(torch.tensor((target_resized[None] == resized_imgs).sum(-1).sum(-1).astype(np.float32)), 0)
        return dict(
            probs=probs,
            idx_k=int(probs.argmax()),
            condition=self.target,
            condition_source=self.raw,
        )


class EdgeSampler:
    def __init__(self, target):
        self.raw = target
        self.sobel_kernel_horizontal = (
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])[
                None, None
            ]
            .requires_grad_(False)
            .cuda()
        )
        self.sobel_kernel_vertical = (
            torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])[
                None, None
            ]
            .requires_grad_(False)
            .cuda()
        )

        self.target = self.sobel_operator(target[None])
        self.cache = {self.target.shape: self.target}

    def sobel_operator(self, img, h=None):
        grey = img.mean(-3, keepdim=True)
        if h is None:
            h, w = grey.shape[-2:]

        output_horizontal = F.conv2d(grey, self.sobel_kernel_horizontal)
        output_vertical = F.conv2d(grey, self.sobel_kernel_vertical)

        magnitude = torch.sqrt(output_horizontal**2 + output_vertical**2)
        thre = 1.5
        thre = 1.0
        edge = magnitude
        if h < 50:
            thre = 2
        if h < 20:
            thre = 2.1
        # edge = (magnitude > thre).float() * 2 - 1
        return edge

    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        # if h < 5:
        #     return dict(probs=torch.rand(k))
        hh, ww = self.target.shape[-2:]
        target_resized = self.target
        if (h, w) not in self.cache and h > 600:

            # raw_resized = nn.functional.interpolate(
            #     self.raw[None], (h, w), mode="area"
            # )
            # target_resized = self.sobel_operator(raw_resized)

            # self.cache[(h, w)] = target_resized
            show(target_resized, rgbs[0], t2rgb, self.sobel_operator(rgbs[:1]))

        # target_resized = self.cache[(h, w)]
        rgbs = nn.functional.interpolate(rgbs, (hh + 2, ww + 2), mode="bilinear")
        resized_imgs = self.sobel_operator(rgbs, h=h)

        if 0:
            # 定义一个3x3的全一矩阵作为卷积核，大小为9
            kernel = torch.ones((1, 1, 3, 3), device="cuda:0")
            # kernel = torch.ones((1, 1, 3, 3), device='cuda:0')

            # 在 target_resized 上执行卷积操作，计算出每个位置的周围有多少个值等于其自身的值
            convolved_imgs = F.conv2d(resized_imgs, kernel, padding=1)

            # 将 target_resized 扩展为与 convolved_imgs 相同的维度以便比较
            target_resized_expanded = target_resized.expand_as(convolved_imgs)

            # 检查卷积后的图像中哪些位置的值等于 target_resized 中的对应值
            masks = convolved_imgs == target_resized_expanded

        # masks = target_resized == resized_imgs
        # probs = (masks).float().mean(-1).mean(-1).mean(-1)
        # probs=resized_imgs[:,:,target_resized[0,0]>0].mean(-1) + (1 - resized_imgs[:,:,target_resized[0,0]<0].mean(-1))
        # probs=  probs[:,0]
        probs = (
            -(torch.abs(resized_imgs - target_resized).clip(0.0))
            .float()
            .mean(-1)
            .mean(-1)
            .mean(-1)
        )
        # probs = - probs
        idx_k = int(probs.argmax())
        g()
        if not increase("edge show") % 8:
            show(resized_imgs[:1], rgbs[:1], t2rgb)
        return dict(
            probs=probs,
            idx_k=idx_k,
            condition=self.target,
            condition_source=resized_imgs[0],
        )


def tensor_to_imagenet_format(rgbs):
    img_tensor = (rgbs + 1) / 2
    # img_tensor = torch.nn.functional.interpolate(img_tensor, size=(224, 224), mode="nearest")
    img_tensor = torch.nn.functional.interpolate(
        img_tensor, size=(64, 64), mode="nearest"
    )

    def normalize(tensor, mean, std):
        return (tensor - mean) / std

    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).cuda()
    img_tensor_norm = normalize(img_tensor, mean, std)
    return img_tensor_norm


class StyleResNet(torchvision.models.ResNet):
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        b, c, h, w = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x.view(b, -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class StyleTransfer:
    def __init__(self, target):

        pretrain = torchvision.models.resnet18(
            weights=torchvision.models.resnet.ResNet18_Weights.DEFAULT
        ).state_dict()
        self.model = StyleResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
        self.model.load_state_dict(pretrain)
        self.model = self.model.eval().cuda()
        self.raw = target
        self.target = self.f(target[None])

    def f(self, rgbs):
        with torch.no_grad():
            return self.model(tensor_to_imagenet_format(rgbs))

    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        feats = self.f(rgbs)
        probs = nn.functional.softmax(-((self.target - feats) ** 2).mean([-1]), 0)
        return dict(
            probs=probs,
            idx_k=int(probs.argmax()),
            condition=self.target,
            condition_source=self.raw,
        )


def topk_sample(probs, topk):
    if isinstance(topk, float):
        topk = max(1, int(round(len(probs) * topk)))
    args = torch.argsort(
        probs,
    )
    idx_k = int(random.choice(args[-topk:]))
    return idx_k


class CifarSampler:
    def __init__(self, target=None):
        with impt("../../asset/resnet_cifar_zhk"):
            from cifar_pretrain import CifarPretrain
        self.model = CifarPretrain()  # rgbs => (k, 10) feat
        self.target = target

    def __call__(self, dic):
        rgbs = dic["rgbs"]
        k, c, h, w = rgbs.shape
        # rgbs = torch.nn.functional.interpolate(rgbs, size=(32, 32), mode="bilinear")
        rgbs = torch.nn.functional.interpolate(rgbs, size=(32, 32), mode="nearest")
        class_probs = nn.functional.softmax(self.model(rgbs), -1)
        target = self.target
        if target is None:
            if "idx_gen" in dic:
                target = int(dic["idx_gen"] % 10)
                # print(target, dic["idx_gen"])
        if target is None:
            probs = class_probs.max(1)[0]
        else:
            probs = class_probs[:, target]

        idx_k = topk_sample(probs, 0.1)
        mg()
        # if random.random() < 1.5 and h<=9:
        # if random.random() < 1.5 and h<=7:
        #     idx_k = random.randint(0, len(rgbs)-1)
        # if not increase("edge show") % 28:
        #     show(rgbs[:1], t2rgb)
        # print(target, dic["idx_gen"])
        # print(classi2name[target],target, class_probs.max(1)[1][:5])
        return dict(
            probs=probs,
            idx_k=idx_k,
            condition=target,
        )


class BatchedGuidedSampler:
    def __init__(self, sampler=None):
        sampler = sampler or (lambda d: dict(probs=[1.0] + [0] * (len(d["rgbs"]) - 1)))
        self.sampler = sampler

    def __call__(self, d):
        outputs = d["output"]
        b, k, c, h, w = outputs.shape
        dics = []
        for batchi in range(b):
            input_dic = dict(rgbs=outputs[batchi])
            if "idx_gens" in d:
                input_dic["idx_gen"] = d["idx_gens"][batchi]
            dic = self.sampler(input_dic)
            dics.append(dic)
        if "idx_k" in dic:
            idx_ks = [dic["idx_k"] for dic in dics]
        elif "probs" in dic:
            idx_ks = npa([np.argmax(dic["probs"]) for dic in dics])
        if "condition_source" in dic:
            d["condition_source"] = dic["condition_source"]
        if "condition" in dic:
            d["condition"] = dic["condition"]
        return npa(idx_ks)


classi2name = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}
if __name__ == "__main__":

    datapath = "../../asset/outputs-cifar.pt"
    # datapath = "../../asset/outputs-ffhq64.pt"
    train_imgs = torch.load(datapath).cuda()

    pklp = "../../asset/v15_00022-cifar10-blockn32_outputk64_chain.dropout0.05_fp32-shot-200000.pkl"
    pklp = "../../asset/v15-00035-cifar10-32x32-cifar_blockn32_outputk64_chain.dropout0.05_fp32_goon.v15.22-shot-087808.pkl"

    # pklp = "../../asset/v15-00040-cifar10-32x32-cifar_fp32_blockn32_outputk512_chain.dropout0.05_transfer.00035.kimg100352-shot-007526.pkl"

    if "ffhq" in datapath:
        pklp = "../../asset/v13_new.setting-00000-ffhq64-fp16-dropout0-200000.pkl"
        pklp = "../../asset/v15-00023-ffhq-64x64-blockn64_outputk64_chain.dropout0.05-shot-117913.pkl"
        # pklp = "../../asset/v17-00016-ffhq-64x64-outputk2_blockn64_chain.dropout0-shot-082790.pkl"
        pklp = "../../asset/v15-00018-ffhq-64x64-blockn64_outputk512_chain.dropout0.05-shot-117913.pkl"

    net = sys._getframe(3).f_globals.get("net")
    if net is None:
        print("load_net")
        net = load_net(pklp)

    train_imgs_xflip = tht(npa(train_imgs)[..., ::-1].copy()).cuda()
    train_idx = 1
    if 1:
        pass
    for train_idx in range(3):
        target = train_imgs[train_idx]
        # target = train_imgs_xflip[train_idx]
        # save_data(d["predict"], "/tmp/predict-as-target")
        # target = load_data("/tmp/predict-as-target")[0]
        dt = dict(target=target[None])

        # sampler = L2Sampler(target)
        # sampler = L1Sampler(target)
        # sampler = ColorfulSampler(target)
        # sampler = SuperResSampler(target, .25)
        # sampler = NoiseSampler(target)
        # sampler = LowBitSampler(target)
        # maski = 8
        # for maski in range(7):
        # sampler = L2MaskedSampler(target, maski)
        # sampler = CannySampler(target, )
        # sampler = EdgeSampler(target, )
        # sampler = StyleTransfer(target, )
        classi = None
        classi = 8

        # for classi in range(10):
        sampler = CifarSampler(classi)

        batch_sampler = BatchedGuidedSampler(sampler)
        d = dict(sampler=batch_sampler)
        d = net(d)
        # tree-d

        print(
            classi,
            classi2name.get(classi, "None"),
            "class:",
            sampler.model(d["predict"]).max(1)[1].item(),
        )

        showd(d, 1, figsize=(8, 5) if target.shape[-1] >= 40 else (4, 3))

        # d=net(dict(batch_size=6));showd(d,1);sampler.model(d["predict"]).max(1)
