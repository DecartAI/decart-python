import asyncio
import os
from pathlib import Path
from decart import DecartClient, models


async def main() -> None:
    async with DecartClient(api_key=os.getenv("DECART_API_KEY", "your-api-key-here")) as client:
        # Image-to-image editing
        image_path = Path(__file__).parent / "files" / "image.png"

        if not image_path.exists():
            print(f"Missing bundled example image at: {image_path}")
            return

        print("Editing image...")
        result = await client.process(
            {
                "model": models.image("lucy-pro-i2i"),
                "prompt": "Apply an impressionist oil-painting treatment while keeping the framing intact",
                "data": image_path,
                "enhance_prompt": True,
            }
        )

        with open("output_i2i.png", "wb") as f:
            f.write(result)

        print("Image saved to output_i2i.png")


if __name__ == "__main__":
    asyncio.run(main())
