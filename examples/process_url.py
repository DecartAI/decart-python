import asyncio
import os
from decart import DecartClient, models


async def main() -> None:
    async with DecartClient(api_key=os.getenv("DECART_API_KEY", "your-api-key-here")) as client:
        print("Transforming video from URL...")
        result = await client.process(
            {
                "model": models.video("lucy-pro-v2v"),
                "prompt": "Watercolor painting style",
                "data": "https://docs.platform.decart.ai/assets/example-video.mp4",
            }
        )

        with open("output_url.mp4", "wb") as f:
            f.write(result)

        print("Video saved to output_url.mp4")


if __name__ == "__main__":
    asyncio.run(main())
