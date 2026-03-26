import asyncio
import os
from pathlib import Path
from decart import DecartClient, models


async def main() -> None:
    # Load the bundled example image
    image_path = Path(__file__).parent / "files" / "image.png"

    if not image_path.exists():
        print(f"Missing bundled example image at: {image_path}")
        return

    async with DecartClient(api_key=os.getenv("DECART_API_KEY", "your-api-key-here")) as client:
        print(f"Loading image: {image_path}")

        # Manual polling - submit and poll yourself
        print("Submitting image-to-motion job...")
        job = await client.queue.submit(
            {
                "model": models.video("lucy-motion"),
                "data": image_path,
                "resolution": "480p",
                "trajectory": [
                    {"frame": 0, "x": 0.35, "y": 0.5},
                    {"frame": 30, "x": 0.5, "y": 0.45},
                    {"frame": 60, "x": 0.65, "y": 0.5},
                ],
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
            with open("output_motion.mp4", "wb") as f:
                f.write(data)
            print("Video saved to output_motion.mp4")
        else:
            print("Job failed")


if __name__ == "__main__":
    asyncio.run(main())
