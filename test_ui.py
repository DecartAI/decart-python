#!/usr/bin/env python3
"""
Decart SDK Test UI - Interactive testing interface for the Python SDK.

Usage:
    pip install gradio
    python test_ui.py

Then open http://localhost:7860 in your browser.
"""

import asyncio
import gradio as gr
from pathlib import Path
from typing import Optional
import tempfile
import os

# Import the SDK
from decart import DecartClient, models


def get_client(api_key: str) -> DecartClient:
    """Create a Decart client with the given API key."""
    if not api_key or not api_key.strip():
        raise ValueError("Please enter an API key")
    return DecartClient(api_key=api_key.strip())


# ============================================================================
# Image Processing (Process API)
# ============================================================================


async def process_text_to_image(
    api_key: str,
    prompt: str,
    seed: Optional[int],
    resolution: str,
    orientation: str,
) -> tuple[Optional[bytes], str]:
    """Generate an image from text prompt."""
    try:
        client = get_client(api_key)

        options = {
            "model": models.image("lucy-pro-t2i"),
            "prompt": prompt,
        }
        if seed:
            options["seed"] = seed
        if resolution and resolution != "default":
            options["resolution"] = resolution
        if orientation and orientation != "default":
            options["orientation"] = orientation

        result = await client.process(options)
        return result, f"Success! Generated image from prompt: '{prompt[:50]}...'"
    except Exception as e:
        return None, f"Error: {str(e)}"


async def process_image_to_image(
    api_key: str,
    prompt: str,
    input_image: str,
    seed: Optional[int],
    strength: float,
) -> tuple[Optional[bytes], str]:
    """Transform an image with a prompt."""
    try:
        if not input_image:
            return None, "Please upload an image"

        client = get_client(api_key)

        options = {
            "model": models.image("lucy-pro-i2i"),
            "prompt": prompt,
            "data": Path(input_image),
        }
        if seed:
            options["seed"] = seed
        if strength:
            options["strength"] = strength

        result = await client.process(options)
        return result, f"Success! Transformed image with prompt: '{prompt[:50]}...'"
    except Exception as e:
        return None, f"Error: {str(e)}"


# ============================================================================
# Video Processing (Queue API)
# ============================================================================


async def process_video_t2v(
    api_key: str,
    prompt: str,
    seed: Optional[int],
    enhance_prompt: bool,
    progress=gr.Progress(),
) -> tuple[Optional[str], str]:
    """Generate a video from text prompt."""
    try:
        client = get_client(api_key)

        options = {
            "model": models.video("lucy-pro-t2v"),
            "prompt": prompt,
        }
        if seed:
            options["seed"] = seed
        if enhance_prompt is not None:
            options["enhance_prompt"] = enhance_prompt

        progress(0.1, desc="Submitting job...")

        def on_status_change(job):
            if job.status == "pending":
                progress(0.2, desc="Job pending...")
            elif job.status == "processing":
                progress(0.5, desc="Processing video...")

        options["on_status_change"] = on_status_change

        result = await client.queue.submit_and_poll(options)

        if result.status == "failed":
            return None, f"Job failed: {result.error}"

        progress(0.9, desc="Saving video...")

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(result.data)
            return f.name, f"Success! Generated video from prompt: '{prompt[:50]}...'"

    except Exception as e:
        return None, f"Error: {str(e)}"


async def process_video_v2v(
    api_key: str,
    prompt: str,
    input_video: str,
    seed: Optional[int],
    enhance_prompt: bool,
    progress=gr.Progress(),
) -> tuple[Optional[str], str]:
    """Transform a video with a prompt."""
    try:
        if not input_video:
            return None, "Please upload a video"

        client = get_client(api_key)

        options = {
            "model": models.video("lucy-pro-v2v"),
            "prompt": prompt,
            "data": Path(input_video),
        }
        if seed:
            options["seed"] = seed
        if enhance_prompt is not None:
            options["enhance_prompt"] = enhance_prompt

        progress(0.1, desc="Submitting job...")

        def on_status_change(job):
            if job.status == "pending":
                progress(0.2, desc="Job pending...")
            elif job.status == "processing":
                progress(0.5, desc="Processing video...")

        options["on_status_change"] = on_status_change

        result = await client.queue.submit_and_poll(options)

        if result.status == "failed":
            return None, f"Job failed: {result.error}"

        progress(0.9, desc="Saving video...")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(result.data)
            return f.name, f"Success! Transformed video with prompt: '{prompt[:50]}...'"

    except Exception as e:
        return None, f"Error: {str(e)}"


async def process_video_restyle(
    api_key: str,
    input_video: str,
    use_reference_image: bool,
    prompt: str,
    reference_image: Optional[str],
    seed: Optional[int],
    enhance_prompt: bool,
    progress=gr.Progress(),
) -> tuple[Optional[str], str]:
    """Restyle a video with prompt OR reference image."""
    try:
        if not input_video:
            return None, "Please upload a video"

        client = get_client(api_key)

        options = {
            "model": models.video("lucy-restyle-v2v"),
            "data": Path(input_video),
        }

        if use_reference_image:
            if not reference_image:
                return None, "Please upload a reference image"
            options["reference_image"] = Path(reference_image)
        else:
            if not prompt:
                return None, "Please enter a prompt"
            options["prompt"] = prompt
            if enhance_prompt is not None:
                options["enhance_prompt"] = enhance_prompt

        if seed:
            options["seed"] = seed

        progress(0.1, desc="Submitting job...")

        def on_status_change(job):
            if job.status == "pending":
                progress(0.2, desc="Job pending...")
            elif job.status == "processing":
                progress(0.5, desc="Processing video...")

        options["on_status_change"] = on_status_change

        result = await client.queue.submit_and_poll(options)

        if result.status == "failed":
            return None, f"Job failed: {result.error}"

        progress(0.9, desc="Saving video...")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(result.data)
            mode = "reference image" if use_reference_image else f"prompt: '{prompt[:30]}...'"
            return f.name, f"Success! Restyled video with {mode}"

    except Exception as e:
        return None, f"Error: {str(e)}"


async def process_video_i2v(
    api_key: str,
    prompt: str,
    input_image: str,
    seed: Optional[int],
    enhance_prompt: bool,
    progress=gr.Progress(),
) -> tuple[Optional[str], str]:
    """Generate a video from an image."""
    try:
        if not input_image:
            return None, "Please upload an image"

        client = get_client(api_key)

        options = {
            "model": models.video("lucy-pro-i2v"),
            "prompt": prompt,
            "data": Path(input_image),
        }
        if seed:
            options["seed"] = seed
        if enhance_prompt is not None:
            options["enhance_prompt"] = enhance_prompt

        progress(0.1, desc="Submitting job...")

        def on_status_change(job):
            if job.status == "pending":
                progress(0.2, desc="Job pending...")
            elif job.status == "processing":
                progress(0.5, desc="Processing video...")

        options["on_status_change"] = on_status_change

        result = await client.queue.submit_and_poll(options)

        if result.status == "failed":
            return None, f"Job failed: {result.error}"

        progress(0.9, desc="Saving video...")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(result.data)
            return f.name, f"Success! Generated video from image"

    except Exception as e:
        return None, f"Error: {str(e)}"


# ============================================================================
# Tokens API
# ============================================================================


async def create_token(api_key: str) -> str:
    """Create a short-lived client token."""
    try:
        client = get_client(api_key)
        result = await client.tokens.create()
        return f"Success!\n\nToken: {result.api_key}\nExpires: {result.expires_at}"
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# Gradio UI
# ============================================================================


def create_ui():
    """Create the Gradio interface."""

    with gr.Blocks(
        title="Decart SDK Test UI",
        theme=gr.themes.Soft(),
        css="""
        .status-success { color: green; }
        .status-error { color: red; }
        """,
    ) as demo:
        gr.Markdown(
            """
        # Decart SDK Test UI

        Interactive testing interface for the Decart Python SDK.
        Enter your API key below to get started.
        """
        )

        # API Key input (shared across all tabs)
        api_key = gr.Textbox(
            label="API Key",
            placeholder="Enter your Decart API key",
            type="password",
            elem_id="api-key-input",
        )

        with gr.Tabs():
            # ================================================================
            # Image Processing Tab
            # ================================================================
            with gr.TabItem("Image Generation"):
                gr.Markdown("### Text to Image")
                with gr.Row():
                    with gr.Column():
                        t2i_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A beautiful sunset over mountains",
                            lines=3,
                        )
                        with gr.Row():
                            t2i_seed = gr.Number(label="Seed (optional)", precision=0)
                            t2i_resolution = gr.Dropdown(
                                label="Resolution",
                                choices=["default", "720p", "1080p"],
                                value="default",
                            )
                            t2i_orientation = gr.Dropdown(
                                label="Orientation",
                                choices=["default", "landscape", "portrait", "square"],
                                value="default",
                            )
                        t2i_btn = gr.Button("Generate Image", variant="primary")
                    with gr.Column():
                        t2i_output = gr.Image(label="Generated Image", type="filepath")
                        t2i_status = gr.Textbox(label="Status", interactive=False)

                t2i_btn.click(
                    fn=lambda *args: asyncio.run(process_text_to_image(*args)),
                    inputs=[api_key, t2i_prompt, t2i_seed, t2i_resolution, t2i_orientation],
                    outputs=[t2i_output, t2i_status],
                )

                gr.Markdown("---")
                gr.Markdown("### Image to Image")
                with gr.Row():
                    with gr.Column():
                        i2i_input = gr.Image(label="Input Image", type="filepath")
                        i2i_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Make it look like anime",
                            lines=2,
                        )
                        with gr.Row():
                            i2i_seed = gr.Number(label="Seed (optional)", precision=0)
                            i2i_strength = gr.Slider(
                                label="Strength",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.75,
                                step=0.05,
                            )
                        i2i_btn = gr.Button("Transform Image", variant="primary")
                    with gr.Column():
                        i2i_output = gr.Image(label="Transformed Image", type="filepath")
                        i2i_status = gr.Textbox(label="Status", interactive=False)

                i2i_btn.click(
                    fn=lambda *args: asyncio.run(process_image_to_image(*args)),
                    inputs=[api_key, i2i_prompt, i2i_input, i2i_seed, i2i_strength],
                    outputs=[i2i_output, i2i_status],
                )

            # ================================================================
            # Video Processing Tab
            # ================================================================
            with gr.TabItem("Video Generation"):
                gr.Markdown("### Text to Video")
                with gr.Row():
                    with gr.Column():
                        t2v_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A cat walking in a park",
                            lines=3,
                        )
                        with gr.Row():
                            t2v_seed = gr.Number(label="Seed (optional)", precision=0)
                            t2v_enhance = gr.Checkbox(label="Enhance Prompt", value=True)
                        t2v_btn = gr.Button("Generate Video", variant="primary")
                    with gr.Column():
                        t2v_output = gr.Video(label="Generated Video")
                        t2v_status = gr.Textbox(label="Status", interactive=False)

                t2v_btn.click(
                    fn=lambda *args: asyncio.run(process_video_t2v(*args)),
                    inputs=[api_key, t2v_prompt, t2v_seed, t2v_enhance],
                    outputs=[t2v_output, t2v_status],
                )

                gr.Markdown("---")
                gr.Markdown("### Image to Video")
                with gr.Row():
                    with gr.Column():
                        i2v_input = gr.Image(label="Input Image", type="filepath")
                        i2v_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="The scene comes to life",
                            lines=2,
                        )
                        with gr.Row():
                            i2v_seed = gr.Number(label="Seed (optional)", precision=0)
                            i2v_enhance = gr.Checkbox(label="Enhance Prompt", value=True)
                        i2v_btn = gr.Button("Generate Video", variant="primary")
                    with gr.Column():
                        i2v_output = gr.Video(label="Generated Video")
                        i2v_status = gr.Textbox(label="Status", interactive=False)

                i2v_btn.click(
                    fn=lambda *args: asyncio.run(process_video_i2v(*args)),
                    inputs=[api_key, i2v_prompt, i2v_input, i2v_seed, i2v_enhance],
                    outputs=[i2v_output, i2v_status],
                )

                gr.Markdown("---")
                gr.Markdown("### Video to Video")
                with gr.Row():
                    with gr.Column():
                        v2v_input = gr.Video(label="Input Video")
                        v2v_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Make it look like Lego world",
                            lines=2,
                        )
                        with gr.Row():
                            v2v_seed = gr.Number(label="Seed (optional)", precision=0)
                            v2v_enhance = gr.Checkbox(label="Enhance Prompt", value=True)
                        v2v_btn = gr.Button("Transform Video", variant="primary")
                    with gr.Column():
                        v2v_output = gr.Video(label="Transformed Video")
                        v2v_status = gr.Textbox(label="Status", interactive=False)

                v2v_btn.click(
                    fn=lambda *args: asyncio.run(process_video_v2v(*args)),
                    inputs=[api_key, v2v_prompt, v2v_input, v2v_seed, v2v_enhance],
                    outputs=[v2v_output, v2v_status],
                )

            # ================================================================
            # Video Restyle Tab (NEW - with reference image support)
            # ================================================================
            with gr.TabItem("Video Restyle (NEW)"):
                gr.Markdown(
                    """
                ### Video Restyle with Prompt OR Reference Image

                This model supports two modes:
                - **Text Prompt**: Describe the style you want
                - **Reference Image**: Upload an image to use as style reference
                """
                )

                with gr.Row():
                    with gr.Column():
                        restyle_input = gr.Video(label="Input Video")
                        restyle_mode = gr.Checkbox(
                            label="Use Reference Image (instead of text prompt)",
                            value=False,
                        )
                        restyle_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Make it look like anime",
                            lines=2,
                            visible=True,
                        )
                        restyle_ref_image = gr.Image(
                            label="Reference Image",
                            type="filepath",
                            visible=False,
                        )
                        with gr.Row():
                            restyle_seed = gr.Number(label="Seed (optional)", precision=0)
                            restyle_enhance = gr.Checkbox(
                                label="Enhance Prompt",
                                value=True,
                                visible=True,
                            )
                        restyle_btn = gr.Button("Restyle Video", variant="primary")
                    with gr.Column():
                        restyle_output = gr.Video(label="Restyled Video")
                        restyle_status = gr.Textbox(label="Status", interactive=False)

                # Toggle visibility based on mode
                def toggle_mode(use_ref):
                    return (
                        gr.update(visible=not use_ref),  # prompt
                        gr.update(visible=use_ref),  # ref image
                        gr.update(visible=not use_ref),  # enhance
                    )

                restyle_mode.change(
                    fn=toggle_mode,
                    inputs=[restyle_mode],
                    outputs=[restyle_prompt, restyle_ref_image, restyle_enhance],
                )

                restyle_btn.click(
                    fn=lambda *args: asyncio.run(process_video_restyle(*args)),
                    inputs=[
                        api_key,
                        restyle_input,
                        restyle_mode,
                        restyle_prompt,
                        restyle_ref_image,
                        restyle_seed,
                        restyle_enhance,
                    ],
                    outputs=[restyle_output, restyle_status],
                )

            # ================================================================
            # Tokens Tab
            # ================================================================
            with gr.TabItem("Tokens"):
                gr.Markdown(
                    """
                ### Create Client Token

                Create a short-lived token for client-side use.
                These tokens are meant for temporary access and expire automatically.
                """
                )

                with gr.Row():
                    with gr.Column():
                        token_btn = gr.Button("Create Token", variant="primary")
                    with gr.Column():
                        token_output = gr.Textbox(
                            label="Result",
                            lines=5,
                            interactive=False,
                        )

                token_btn.click(
                    fn=lambda key: asyncio.run(create_token(key)),
                    inputs=[api_key],
                    outputs=[token_output],
                )

        gr.Markdown(
            """
        ---
        **Note**: This UI uses the Decart Python SDK.
        For realtime/WebRTC features, use the example scripts in `examples/`.
        """
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1", #localhost only
        server_port=7860,
        share=False,
    )
