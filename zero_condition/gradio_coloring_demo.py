#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import boxx
from boxx import *
import time
import numpy as np
import gradio as gr
from condition256_gen import DDNInference
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


def generate(input_condition_img, editor_value):
    tree([input_condition_img, editor_value])
    ddn = get_model()
    d = ddn.coloring_demo_inference(
        input_condition_img, n_samples=n_samples, guided_rgba=editor_value["composite"]
    )
    stage_last_predicts = d["stage_last_predicts_np"]
    tree(stage_last_predicts)
    return flatten_results(stage_last_predicts)


def input_condition_img_callback(input_condition_img, editor_value):
    if isinstance(input_condition_img, np.ndarray):
        if input_condition_img.shape[0] != H or input_condition_img.shape[1] != W:
            input_condition_img = resize(input_condition_img, (H, W))
    editor_value["background"] = np.uint8(input_condition_img.mean(-1).round())
    return editor_value


def flatten_results(results):
    return sum([results[key] for key in sorted(results)], [])


with gr.Blocks(css=".gr-box{padding:0!important}") as demo:
    with gr.Row():
        with gr.Row():
            upload_block = gr.Image(
                label="input_condition_img",
                width=256,
                height=256,
            )
            editor_block = gr.ImageEditor(
                label="ZSCG condition",
                type="numpy",
                crop_size="1:1",
                width=256,
                height=256,
                # canvas_size=(256, 256),   # 画布固定为 256×256
                # fixed_canvas=True,        # 上传 / 更换 background 时自动缩放到画布大小
            )
            prompt_block = gr.Textbox(
                label="CLIP prompt for Zero-Shot-Conditional-Generation:"
            )

    button = gr.Button("generate")
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
    n_samples = 3
    results = {}
    gr.Markdown("## Results")
    gr.Markdown("Each column is a sample, each row is last predict of a stage")
    with gr.Row():
        for sample_idx in range(n_samples):
            with gr.Column():
                for stage_idx in range(9)[::-1]:
                    size = 2**stage_idx
                    key = f"{size}x{size}"
                    result_image = gr.Image(
                        width=max(80, size + 16), height=size + 2, **clean_img_args
                    )
                    results[key] = results.get(key, []) + [result_image]
    button.click(
        generate, inputs=[upload_block, editor_block], outputs=flatten_results(results)
    )

    example_rgb = boxx.imread(
        "/home/yl/dataset/ffhq/test_self/test_self/FFHQ-test4.png"
    )
    example_img = np.uint8(
        example_rgb.mean(
            -1,
        )
    )
    gr.HTML("<hr>")
    with gr.Row() as example1:
        with gr.Column():
            example_img_gr1 = gr.Image(
                example_img, width=64 + 16, height=64 + 2, **clean_img_args
            )
        with gr.Column():
            gr.Markdown("example1: Musk\n\nprompt: None")

    def load_example(example, prompt):
        example

    example1

    gr.HTML("<hr>")

if __name__ == "__main__":
    demo.launch(debug=True, server_name="0.0.0.0")
