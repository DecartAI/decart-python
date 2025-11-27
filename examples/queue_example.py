import asyncio
import os
from decart import DecartClient, models


async def main() -> None:
    async with DecartClient(api_key=os.getenv("DECART_API_KEY", "your-api-key-here")) as client:
        # Automatic polling - submits and waits for completion
        print("Submitting job with automatic polling...")
        result = await client.queue.submit_and_poll(
            {
                "model": models.video("lucy-pro-t2v"),
                "prompt": "A serene lake at sunset with mountains in the background",
                "resolution": "480p",
                "on_status_change": lambda job: print(f"Job {job.job_id}: {job.status}"),
            }
        )

        if result.status == "completed":
            with open("output_queue_auto.mp4", "wb") as f:
                f.write(result.data)
            print("Video saved to output_queue_auto.mp4")
        else:
            print(f"Job failed: {result.error}")

        # Manual polling - submit and poll yourself
        print("\nSubmitting job with manual polling...")
        job = await client.queue.submit(
            {
                "model": models.video("lucy-pro-t2v"),
                "prompt": "A cat playing piano in a cozy living room",
                "resolution": "480p",
            }
        )
        print(f"Job submitted: {job.job_id}")

        status = await client.queue.status(job.job_id)
        while status.status in ("pending", "processing"):
            print(f"Status: {status.status}")
            await asyncio.sleep(2)
            status = await client.queue.status(job.job_id)

        if status.status == "completed":
            data = await client.queue.result(job.job_id)
            with open("output_queue_manual.mp4", "wb") as f:
                f.write(data)
            print("Video saved to output_queue_manual.mp4")
        else:
            print("Job failed")


if __name__ == "__main__":
    asyncio.run(main())
