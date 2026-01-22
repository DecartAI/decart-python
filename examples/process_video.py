"""
Video generation example using the Queue API.
Video models only support async queue processing.
"""

import asyncio
import os
from decart import DecartClient, models


async def main() -> None:
    async with DecartClient(api_key=os.getenv("DECART_API_KEY", "your-api-key-here")) as client:
        # Text-to-video generation
        print("Generating video from text...")
        result = await client.queue.submit_and_poll(
            {
                "model": models.video("lucy-pro-t2v"),
                "prompt": "A serene lake at sunset with mountains in the background",
                "seed": 42,
                "on_status_change": lambda job: print(f"  Status: {job.status}"),
            }
        )

        if result.status == "completed":
            with open("output_t2v.mp4", "wb") as f:
                f.write(result.data)
            print("Video saved to output_t2v.mp4")
        else:
            print(f"Text-to-video failed: {result.error}")
            return

        # Video-to-video transformation
        print("\nTransforming video...")
        with open("output_t2v.mp4", "rb") as video_file:
            result = await client.queue.submit_and_poll(
                {
                    "model": models.video("lucy-pro-v2v"),
                    "prompt": "Anime style with vibrant colors",
                    "data": video_file,
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
