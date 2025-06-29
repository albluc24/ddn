#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import boxx
from boxx import *
import time
import numpy as np
import gradio as gr
from condition256_gen import DDNInference, crop_and_resize
import threading

H, W = 256, 256

ddn = None


def get_model():
    with threading.Lock():
        global ddn
        if ddn is None:
            ddn = DDNInference(
                "../../asset/v32-00001-ffhq-256x256-ffhq256_cond.color_chain.dropout0.05_batch128-shot-200000.pkl"
            )
        return ddn


def generate(input_condition_img, editor_value, prompt_value=None):
    tree([input_condition_img, editor_value, prompt_value])
    ddn = get_model()
    d = ddn.coloring_demo_inference(
        input_condition_img,
        n_samples=n_samples,
        guided_rgba=editor_value["layers"][0] if len(editor_value["layers"]) else None,
        clip_prompt=prompt_value,
    )
    stage_last_predicts = d["stage_last_predicts_np"]
    tree(stage_last_predicts)
    return flatten_results(stage_last_predicts)


def input_condition_img_callback(input_condition_img, editor_value):
    if isinstance(input_condition_img, np.ndarray):
        if input_condition_img.shape[0] != H or input_condition_img.shape[1] != W:
            input_condition_img = crop_and_resize(input_condition_img, (H, W))
    if input_condition_img is None:
        editor_value["background"] = None
        editor_value["composite"] = None
    else:
        editor_value["background"] = np.uint8(input_condition_img.mean(-1).round())
        editor_value["composite"] = None
    return editor_value


def flatten_results(results):
    return sum([results[key] for key in sorted(result_blocks)], [])


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Row():
            upload_block = gr.Image(
                label="input_condition_img",
                width=256,
                height=256,
            )
            brush = gr.Brush()
            brush.colors = [
                "rgb(39,106,167)",
                "rgb(251,217,196)",
                "rgb(255,194,116)",
                "rgb(229,220,138)",
                "rgb(203,172,225)",
            ] + brush.colors
            brush.default_size = 8
            brush.default_color = brush.colors[0]
            editor_block = gr.ImageEditor(
                label="RGBA guided",
                type="numpy",
                crop_size="1:1",
                width=256,
                height=256,
                brush=brush,
                layers=False,
                # canvas_size=(256, 256),   # 画布固定为 256×256
                # fixed_canvas=True,        # 上传 / 更换 background 时自动缩放到画布大小
            )
            with gr.Column():
                prompt_block = gr.Textbox(
                    "",
                    interactive=True,
                    label="CLIP prompt for Zero-Shot-Conditional-Generation:",
                )

                button = gr.Button("generate")
    gr.HTML("<hr>")
    upload_block.change(
        input_condition_img_callback,
        inputs=[upload_block, editor_block],
        outputs=editor_block,
    )

    clean_img_args = dict(
        interactive=False,  # 只显示结果
        show_label=False,  # 不显示标题
        show_download_button=False,  # 不显示下载按钮
        min_width=1,
        show_fullscreen_button=False,
        format="png",
        # container=False
    )
    n_samples = 12
    min_size = 256
    result_blocks = {}
    # gr.Markdown("## Results")
    # gr.Markdown("Each column is a sample, each row is last predict of a stage")
    with gr.Row():
        for sample_idx in range(n_samples):
            with gr.Column(min_width=256):
                for stage_idx in range(9)[::-1]:
                    size = 2**stage_idx
                    key = f"{size}x{size}"
                    if size < min_size:
                        continue
                    result_image = gr.Image(
                        width=max(80, size + 16), height=size + 2, **clean_img_args
                    )
                    result_blocks[key] = result_blocks.get(key, []) + [result_image]
    button.click(
        generate,
        inputs=[upload_block, editor_block, prompt_block],
        outputs=flatten_results(result_blocks),
    )

    example_rgb = boxx.imread("../../ddn_asset/ffhq_example/FFHQ-test4.png")
    example_img = np.uint8(
        example_rgb.mean(
            -1,
        )
    )
    if 0:
        gr.HTML("<hr>")
        with gr.Row() as example1:
            with gr.Column():
                example_img_gr1 = gr.Image(
                    example_img, width=64 + 16, height=64 + 2, **clean_img_args
                )
            with gr.Column():
                gr.Markdown("example1: Musk\n\nprompt: None")

    gr.HTML("<hr>")

    guided_rgba = example_rgb[:]
    # guided_rgba[:] = [[255,194,116]]
    # guided_rgba[:] = [[229,220,138]]
    guided_rgba[:] = [[203, 172, 225]]
    mask = np.ones_like(guided_rgba)[..., :1] * 0
    mask[: len(mask) // 10 :, : len(mask) // 10] = 255
    guided_rgba = np.concatenate([guided_rgba, mask], axis=-1)

    editor_value_example = dict(
        background=example_img // 2, layers=[guided_rgba], composite=guided_rgba
    )

    def read_as_input_condition_img(png_path):
        img = np.uint8(boxx.imread(png_path).mean(-1))
        return crop_and_resize(img, (H, W))

    def read_as_example_input(png_path):
        condition_img = read_as_input_condition_img(png_path)
        return [
            condition_img,
            dict(background=condition_img // 2, layers=[], composite=None),
            "",
        ]

    example_inputs = [
        read_as_example_input(png_path)
        for png_path in glob.glob("../../ddn_asset/ffhq_example/*.png")
    ]
    gr.Examples(
        # [[example_img, editor_value_example, "null"]] +
        [
            [
                *read_as_example_input("../../ddn_asset/ffhq_example/FFHQ-test5.png")[
                    :2
                ],
                "portrait with dark red hair",
            ]
        ]
        + example_inputs,
        inputs=[upload_block, editor_block, prompt_block],
        outputs=flatten_results(result_blocks),
        fn=generate,
    )

if __name__ == "__main__":
    demo.launch(debug=True, server_name="0.0.0.0")
