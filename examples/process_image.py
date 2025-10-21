import asyncio
import os
from decart import DecartClient, models


async def main() -> None:
    async with DecartClient(api_key=os.getenv("DECART_API_KEY", "your-api-key-here")) as client:
        print("Generating image from text...")
        result = await client.process(
            {
                "model": models.image("lucy-pro-t2i"),
                "prompt": "A futuristic cityscape at night with neon lights",
                "seed": 42,
                "orientation": "portrait",
            }
        )

        with open("output_t2i.png", "wb") as f:
            f.write(result)

        print("Image saved to output_t2i.png")

        print("\nTransforming image...")
        with open("output_t2i.png", "rb") as image_file:
            result = await client.process(
                {
                    "model": models.image("lucy-pro-i2i"),
                    "prompt": "Oil painting style with impressionist brushstrokes",
                    "data": image_file,
                    "enhance_prompt": True,
                }
            )

        with open("output_i2i.png", "wb") as f:
            f.write(result)

        print("Image saved to output_i2i.png")


if __name__ == "__main__":
    asyncio.run(main())
