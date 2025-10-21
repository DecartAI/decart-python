import asyncio
import logging
import os
import numpy as np
from pathlib import Path
from decart import DecartClient, models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

try:
    from aiortc import VideoStreamTrack
    from aiortc.contrib.media import MediaRecorder
    from av import VideoFrame
except ImportError:
    print("aiortc is required for this example.")
    print("Install with: pip install decart[realtime]")
    exit(1)


class SyntheticVideoTrack(VideoStreamTrack):
    """
    Generates synthetic video frames for testing.
    Creates colored frames that change over time.
    """

    def __init__(self):
        super().__init__()
        self.counter = 0

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
        ]

        color_index = (self.counter // 25) % len(colors)
        color = colors[color_index]

        img = np.zeros((704, 1280, 3), dtype=np.uint8)
        img[:] = color

        frame = VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base

        self.counter += 1

        return frame


async def main():
    api_key = os.getenv("DECART_API_KEY")
    if not api_key:
        print("Error: DECART_API_KEY environment variable not set")
        print("Usage: DECART_API_KEY=your-key python realtime_synthetic.py")
        return

    try:
        from decart.realtime.client import RealtimeClient
    except ImportError:
        print("Error: Realtime API not available")
        print("Install with: pip install decart[realtime]")
        return

    print("Creating Decart client...")
    async with DecartClient(api_key=api_key) as client:
        print("Creating synthetic video track...")
        video_track = SyntheticVideoTrack()

        model = models.realtime("mirage")
        print(f"Using model: {model.name}")
        print(f"Model config - FPS: {model.fps}, Size: {model.width}x{model.height}")

        frame_count = 0
        recorder = None
        output_file = Path("output_realtime_synthetic.mp4")

        def on_remote_stream(track):
            nonlocal frame_count, recorder
            frame_count += 1
            print(f"📹 Received remote stream frame #{frame_count}")

            if recorder is None:
                print(f"💾 Recording to {output_file}")
                recorder = MediaRecorder(str(output_file))
                recorder.addTrack(track)
                asyncio.create_task(recorder.start())

        def on_connection_change(state):
            print(f"🔄 Connection state: {state}")

        def on_error(error):
            print(f"❌ Error: {error.__class__.__name__} - {error.message}")

        print("\nConnecting to Realtime API...")
        try:
            from decart.realtime.client import RealtimeClient
            from decart.realtime.types import RealtimeConnectOptions
            from decart.types import ModelState, Prompt

            realtime_client = await RealtimeClient.connect(
                base_url=client.base_url,
                api_key=client.api_key,
                local_track=video_track,
                options=RealtimeConnectOptions(
                    model=model,
                    on_remote_stream=on_remote_stream,
                    initial_state=ModelState(
                        prompt=Prompt(text="Anime style", enrich=True), mirror=False
                    ),
                ),
            )

            realtime_client.on("connection_change", on_connection_change)
            realtime_client.on("error", on_error)

            print("✓ Connected!")
            print(f"Session ID: {realtime_client.session_id}")
            print("Processing video for 10 seconds...")

            try:
                await asyncio.sleep(5)

                print("\n🎨 Changing style to 'Cyberpunk city'...")
                await realtime_client.set_prompt("Cyberpunk city")

                await asyncio.sleep(5)

                print(f"\n✓ Processed {frame_count} frames total")
            finally:
                if recorder:
                    try:
                        print(f"💾 Saving video to {output_file}...")
                        await asyncio.sleep(0.5)
                        await recorder.stop()
                        print(f"✓ Video saved to {output_file}")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not save video cleanly: {e}")
                        print("   Video file may be incomplete or corrupted")

        except Exception as e:
            print(f"\n❌ Connection failed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            if "realtime_client" in locals():
                print("\nDisconnecting...")
                await realtime_client.disconnect()
                print("✓ Disconnected")


if __name__ == "__main__":
    asyncio.run(main())
