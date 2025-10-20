import asyncio
import os
from decart_sdk import create_decart_client, models


async def main() -> None:
    client = create_decart_client(api_key=os.getenv("DECART_API_KEY", "your-api-key-here"))

    print("Generating video from text...")
    result = await client.process(
        {
            "model": models.video("lucy-pro-t2v"),
            "prompt": "A serene lake at sunset with mountains in the background",
            "seed": 42,
        }
    )

    with open("output_t2v.mp4", "wb") as f:
        f.write(result)

    print("Video saved to output_t2v.mp4")

    print("Transforming video...")
    with open("output_t2v.mp4", "rb") as video_file:
        result = await client.process(
            {
                "model": models.video("lucy-pro-v2v"),
                "prompt": "Anime style with vibrant colors",
                "data": video_file,
                "enhance_prompt": True,
                "num_inference_steps": 50,
            }
        )

    with open("output_v2v.mp4", "wb") as f:
        f.write(result)

    print("Video saved to output_v2v.mp4")


if __name__ == "__main__":
    asyncio.run(main())
