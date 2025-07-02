#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import boxx
from boxx import *
import time
import numpy as np
import gradio as gr
from condition256_gen import DDNInference, crop_and_resize
import threading
from PIL import Image
import io

H, W = 256, 256

ddn = None


def get_model():
    with threading.Lock():
        global ddn
        if ddn is None:
            ddn = DDNInference(
                "../../ddn_asset/v32-00003-ffhq-256x256-ffhq256_cond.color_chain.dropout0.05_batch64_k64-shot-200000.pkl"
            )
        return ddn


def generate(input_condition_img, editor_value, prompt_value=None):
    tree([input_condition_img, editor_value, prompt_value])
    ddn = get_model()
    n = n_samples
    if prompt_value:
        n = n_samples_with_clip
    d = ddn.coloring_demo_inference(
        input_condition_img,
        n_samples=n,
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
        editor_value["background"] = np.uint8(input_condition_img.mean(-1).round()) // 2
        editor_value["composite"] = None
    return editor_value


def flatten_results(results):
    return sum(
        [
            (results[key] + [None] * n_samples)[:n_samples]
            for key in sorted(result_blocks)
        ],
        [],
    )


def read_as_input_condition_img(png_path):
    img = boxx.imread(png_path)
    return crop_and_resize(img, (H, W))


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Row():
            default_rgb = read_as_input_condition_img(
                "../../ddn_asset/ffhq_example/FFHQ-test4.png"
            )
            default_edit = dict(
                background=np.uint8(default_rgb.mean(-1)) // 2,
                layers=[],
                composite=None,
            )
            upload_block = gr.Image(
                default_rgb,
                label="input_condition_img",
                format="png",
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
                "rgb(255,0,0)",
                "rgb(0,255,0)",
                "rgb(255,255,0)",
                "rgb(0,0,255)",
                "rgb(0,255,255)",
            ] + brush.colors
            brush.default_size = 8
            brush.default_color = brush.colors[0]
            editor_block = gr.ImageEditor(
                default_edit,
                label="RGBA guided",
                type="numpy",
                crop_size="1:1",
                width=256,
                height=256,
                brush=brush,
                layers=False,
                format="png",
                # canvas_size=(256, 256),   # 画布固定为 256×256
                # fixed_canvas=True,        # 上传 / 更换 background 时自动缩放到画布大小
            )
            with gr.Column():
                prompt_block = gr.Textbox(
                    "",
                    interactive=True,
                    label="CLIP prompt for Zero-Shot-Conditional-Generation:",
                )
                gr.HTML("<br><br><br>")

                button = gr.Button("generate")
    gr.HTML("<hr style='padding:0!important'>", container=False)
    upload_block.change(
        input_condition_img_callback,
        inputs=[upload_block, editor_block],
        outputs=editor_block,
    )

    clean_img_args = dict(
        interactive=False,
        show_label=False,
        show_download_button=False,
        min_width=1,
        show_fullscreen_button=False,
        format="png",
        # container=False
    )
    n_samples = 12
    n_samples_with_clip = 4
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

    def read_as_example_input(png_path):
        condition_img = read_as_input_condition_img(png_path)
        return [
            condition_img,
            dict(
                background=np.uint8(condition_img.mean(-1)) // 2,
                layers=[],
                composite=None,
            ),
            "",
        ]

    example_inputs = []
    example_inputs.append(
        [
            default_rgb,
            default_edit,
            "",
        ]
    )
    _rgb, _edit = read_as_example_input("../../ddn_asset/ffhq_example/FFHQ-00526.png")[
        :2
    ]
    example_inputs.append(
        [
            np.uint8(_rgb.mean(-1)),
            _edit,
            "",
        ]
    )
    _rgb, _edit = read_as_example_input("../../ddn_asset/ffhq_example/FFHQ-test2.png")[
        :2
    ]
    example_inputs.append(
        [
            np.uint8(_rgb.mean(-1)),
            _edit,
            "colorful portrait",
        ]
    )
    example_inputs.append(
        [
            *read_as_example_input("../../ddn_asset/ffhq_example/FFHQ-test5.png")[:2],
            "portrait with purple hair",
        ]
    )
    # example_inputs += [
    #     read_as_example_input(png_path)
    #     for png_path in sorted(glob.glob("../../ddn_asset/ffhq_example/*.png"))
    # ]
    gr.Examples(
        # [[example_img, editor_value_example, "null"]] +
        example_inputs,
        inputs=[upload_block, editor_block, prompt_block],
        outputs=flatten_results(result_blocks),
        fn=generate,
    )

    gr.HTML("<hr>")

    # GIF area
    def convert_to_gif(*flattened_results):
        """Convert valid result blocks to GIF animation"""
        if not flattened_results or all(img is None for img in flattened_results):
            return None

        # Rebuild result_blocks structure from flattened results
        result_dict = {}
        idx = 0
        for sample_idx in range(n_samples):
            for stage_idx in range(9)[::-1]:
                size = 2**stage_idx
                key = f"{size}x{size}"
                if size < min_size:
                    continue

                if key not in result_dict:
                    result_dict[key] = []

                if idx < len(flattened_results):
                    result_dict[key].append(flattened_results[idx])
                else:
                    result_dict[key].append(None)
                idx += 1

        # Collect valid images, grouped by stages
        stage_images = {}

        for sample_idx in range(n_samples):
            for stage_idx in range(9)[::-1]:
                size = 2**stage_idx
                key = f"{size}x{size}"
                if size < min_size:
                    continue

                if key in result_dict and sample_idx < len(result_dict[key]):
                    result_img = result_dict[key][sample_idx]
                    if result_img is not None:
                        if key not in stage_images:
                            stage_images[key] = []
                        # Convert to PIL Image
                        if isinstance(result_img, np.ndarray):
                            img = Image.fromarray(result_img)
                        else:
                            img = result_img
                        stage_images[key].append(img)

        if not stage_images:
            return None

        # Use largest size images to create GIF
        largest_size = max(stage_images.keys(), key=lambda x: int(x.split("x")[0]))
        frames = stage_images[largest_size]

        if not frames:
            return None

        # Create GIF in memory and return as PIL Image
        gif_buffer = io.BytesIO()
        frames[0].save(
            gif_buffer,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=500,  # 100ms per frame = 10 fps
            loop=0,
        )
        gif_buffer.seek(0)

        # Load the GIF back as PIL Image to preserve animation
        gif_img = Image.open(gif_buffer)
        return gif_img

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Convert Results to GIF Animation")
            convert_gif_button = gr.Button("Convert to GIF", variant="secondary")
        with gr.Column():
            gif_output = gr.Image(
                label="GIF Animation",
                type="pil",
                width=256,
                height=256,
                interactive=False,
                show_download_button=True,
                format="gif",
            )

    # Connect the conversion function to button
    convert_gif_button.click(
        convert_to_gif,
        inputs=flatten_results(result_blocks),  # Use all result_blocks as input
        outputs=gif_output,
    )

if __name__ == "__main__":
    demo.launch(debug=True, server_name="0.0.0.0")
