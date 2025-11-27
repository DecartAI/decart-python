import asyncio
import os
from pathlib import Path
from decart import DecartClient, models


async def main() -> None:
    # Load image from assets folder
    image_path = Path(__file__).parent / "assets" / "example_asset.png"

    if not image_path.exists():
        print(f"Please add an image at: {image_path}")
        return

    async with DecartClient(api_key=os.getenv("DECART_API_KEY", "your-api-key-here")) as client:
        print(f"Loading image: {image_path}")

        # Manual polling - submit and poll yourself
        print("Submitting job...")
        job = await client.queue.submit(
            {
                "model": models.video("lucy-pro-i2v"),
                "prompt": "The image comes to life with gentle motion",
                "data": image_path,
                "resolution": "480p",
            }
        )
        print(f"Job submitted: {job.job_id}")

        # Poll manually
        status = await client.queue.status(job.job_id)
        while status.status in ("pending", "processing"):
            print(f"Status: {status.status}")
            await asyncio.sleep(2)
            status = await client.queue.status(job.job_id)

        print(f"Final status: {status.status}")

        if status.status == "completed":
            print("Fetching result...")
            data = await client.queue.result(job.job_id)
            with open("output_i2v.mp4", "wb") as f:
                f.write(data)
            print("Video saved to output_i2v.mp4")
        else:
            print("Job failed")


if __name__ == "__main__":
    asyncio.run(main())
