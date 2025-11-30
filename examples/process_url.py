"""
Video transformation from URL example using the Queue API.
Video models only support async queue processing.
"""

import asyncio
import os
from decart import DecartClient, models


async def main() -> None:
    async with DecartClient(api_key=os.getenv("DECART_API_KEY", "your-api-key-here")) as client:
        print("Transforming video from URL...")
        result = await client.queue.submit_and_poll(
            {
                "model": models.video("lucy-pro-v2v"),
                "prompt": "Watercolor painting style",
                "data": "https://docs.platform.decart.ai/assets/example-video.mp4",
                "on_status_change": lambda job: print(f"  Status: {job.status}"),
            }
        )

        if result.status == "completed":
            with open("output_url.mp4", "wb") as f:
                f.write(result.data)
            print("Video saved to output_url.mp4")
        else:
            print(f"Job failed: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
