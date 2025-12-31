"""
Video Restyle Example

This example demonstrates how to use the lucy-restyle-v2v model to restyle a video
using either a text prompt OR a reference image.

Usage:
    # With text prompt:
    DECART_API_KEY=your-key python video_restyle.py input.mp4 --prompt "anime style"

    # With reference image:
    DECART_API_KEY=your-key python video_restyle.py input.mp4 --reference style.png

Requirements:
    pip install decart
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path

from decart import DecartClient, models


async def main():
    parser = argparse.ArgumentParser(
        description="Restyle a video using text prompt or reference image"
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument(
        "--prompt",
        "-p",
        help="Text prompt describing the style (e.g., 'anime style', 'oil painting')",
    )
    parser.add_argument("--reference", "-r", help="Path to reference image for style transfer")
    parser.add_argument("--output", "-o", help="Output file path (default: output_restyle.mp4)")
    parser.add_argument("--seed", "-s", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--enhance",
        action="store_true",
        default=True,
        help="Enhance the prompt (only with --prompt, default: True)",
    )
    parser.add_argument("--no-enhance", action="store_true", help="Disable prompt enhancement")

    args = parser.parse_args()

    # Validate arguments
    if not args.prompt and not args.reference:
        print("Error: Must provide either --prompt or --reference")
        parser.print_help()
        sys.exit(1)

    if args.prompt and args.reference:
        print("Error: Cannot use both --prompt and --reference")
        print("       Please choose one or the other")
        sys.exit(1)

    api_key = os.getenv("DECART_API_KEY")
    if not api_key:
        print("Error: DECART_API_KEY environment variable not set")
        sys.exit(1)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    if args.reference:
        ref_path = Path(args.reference)
        if not ref_path.exists():
            print(f"Error: Reference image not found: {ref_path}")
            sys.exit(1)

    output_path = args.output or f"output_restyle_{video_path.stem}.mp4"

    print("=" * 50)
    print("Video Restyle")
    print("=" * 50)
    print(f"Input video: {video_path}")
    if args.prompt:
        print(f"Style: Text prompt - '{args.prompt}'")
        print(f"Enhance prompt: {not args.no_enhance}")
    else:
        print(f"Style: Reference image - {args.reference}")
    print(f"Output: {output_path}")
    if args.seed:
        print(f"Seed: {args.seed}")
    print("=" * 50)

    async with DecartClient(api_key=api_key) as client:
        # Build options
        options = {
            "model": models.video("lucy-restyle-v2v"),
            "data": video_path,
        }

        if args.prompt:
            options["prompt"] = args.prompt
            options["enhance_prompt"] = not args.no_enhance
        else:
            options["reference_image"] = Path(args.reference)

        if args.seed:
            options["seed"] = args.seed

        def on_status_change(job):
            status_emoji = {
                "pending": "⏳",
                "processing": "🔄",
                "completed": "✅",
                "failed": "❌",
            }
            emoji = status_emoji.get(job.status, "•")
            print(f"{emoji} Status: {job.status}")

        options["on_status_change"] = on_status_change

        print("\nSubmitting job...")
        result = await client.queue.submit_and_poll(options)

        if result.status == "failed":
            print(f"\n❌ Job failed: {result.error}")
            sys.exit(1)

        print("\n✅ Job completed!")
        print(f"💾 Saving to {output_path}...")

        with open(output_path, "wb") as f:
            f.write(result.data)

        print(f"✓ Video saved to {output_path}")
        print(f"  Size: {len(result.data) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    asyncio.run(main())
