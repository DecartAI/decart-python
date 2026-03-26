"""
Video editing example using the Queue API.
Video models only support async queue processing.
"""

import asyncio
import os
from pathlib import Path
from decart import DecartClient, models


async def main() -> None:
    async with DecartClient(api_key=os.getenv("DECART_API_KEY", "your-api-key-here")) as client:
        # Video-to-video editing
        video_path = Path(__file__).parent / "assets" / "example_video.mp4"

        if not video_path.exists():
            print(f"Please add a video at: {video_path}")
            return

        print("Editing video...")
        result = await client.queue.submit_and_poll(
            {
                "model": models.video("lucy-pro-v2v"),
                "prompt": "Restyle this footage with anime shading, vibrant highlights, and crisp outlines",
                "data": video_path,
                "enhance_prompt": True,
                "on_status_change": lambda job: print(f"  Status: {job.status}"),
            }
        )

        if result.status == "completed":
            with open("output_v2v.mp4", "wb") as f:
                f.write(result.data)
            print("Video saved to output_v2v.mp4")
        else:
            print(f"Video-to-video failed: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
