"""
Virtual Try-On Example

This example demonstrates how to use the lucy-2.1-vton model to perform
virtual try-on on a video using a reference garment image.

Usage:
    # With reference image and prompt:
    DECART_API_KEY=your-key python video_tryon.py input.mp4 --reference garment.png --prompt "wear this outfit"

    # With reference image only (empty prompt):
    DECART_API_KEY=your-key python video_tryon.py input.mp4 --reference garment.png

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
        description="Virtual try-on: apply a garment from a reference image onto a person in a video"
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument(
        "--reference", "-r", required=True, help="Path to reference garment image"
    )
    parser.add_argument(
        "--prompt", "-p", default="", help="Text prompt (default: empty string)"
    )
    parser.add_argument("--output", "-o", help="Output file path (default: output_tryon.mp4)")
    parser.add_argument("--seed", "-s", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--enhance",
        action="store_true",
        default=True,
        help="Enhance the prompt (default: True)",
    )
    parser.add_argument("--no-enhance", action="store_true", help="Disable prompt enhancement")

    args = parser.parse_args()

    api_key = os.getenv("DECART_API_KEY")
    if not api_key:
        print("Error: DECART_API_KEY environment variable not set")
        sys.exit(1)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"Error: Reference image not found: {ref_path}")
        sys.exit(1)

    output_path = args.output or f"output_tryon_{video_path.stem}.mp4"

    print("=" * 50)
    print("Virtual Try-On")
    print("=" * 50)
    print(f"Input video: {video_path}")
    print(f"Reference image: {ref_path}")
    if args.prompt:
        print(f"Prompt: '{args.prompt}'")
        print(f"Enhance prompt: {not args.no_enhance}")
    print(f"Output: {output_path}")
    if args.seed:
        print(f"Seed: {args.seed}")
    print("=" * 50)

    async with DecartClient(api_key=api_key) as client:
        options = {
            "model": models.video("lucy-2.1-vton"),
            "data": video_path,
            "prompt": args.prompt,
            "reference_image": ref_path,
        }

        if args.prompt:
            options["enhance_prompt"] = not args.no_enhance

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
