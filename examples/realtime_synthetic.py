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
    from livekit import rtc
except ImportError:
    print("livekit is required for this example.")
    print("Install with: pip install decart[realtime]")
    exit(1)


class SyntheticVideoSource:
    """Pushes synthetic RGB frames into a LiveKit video source."""

    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps
        self.source = rtc.VideoSource(width, height)
        self.track = rtc.LocalVideoTrack.create_video_track("synthetic-video", self.source)
        self.counter = 0
        self._running = False

    async def start(self):
        self._running = True
        frame_interval = 1 / self.fps

        while self._running:
            self.source.capture_frame(self._next_frame())
            await asyncio.sleep(frame_interval)

    def stop(self):
        self._running = False

    def _next_frame(self):
        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
        ]

        color_index = (self.counter // 25) % len(colors)
        color = colors[color_index]

        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = color

        self.counter += 1
        return rtc.VideoFrame(
            width=self.width,
            height=self.height,
            type=rtc.VideoBufferType.RGB24,
            data=img.tobytes(),
        )


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
        model = models.realtime("lucy-2.1")
        video_source = SyntheticVideoSource(model.width, model.height, model.fps)
        print(f"Using model: {model.name}")
        print(f"Model config - FPS: {model.fps}, Size: {model.width}x{model.height}")

        frame_count = 0
        source_task = asyncio.create_task(video_source.start())
        output_file = Path("output_realtime_synthetic.frames")

        def on_remote_stream(track):
            print(f"📹 Received remote LiveKit track: {track.sid}")

            async def consume_frames():
                nonlocal frame_count
                async for event in rtc.VideoStream(track):
                    frame_count += 1
                    if frame_count % 25 == 0:
                        print(f"📹 Received {frame_count} remote frames")

            asyncio.create_task(consume_frames())

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
                base_url=client.realtime_base_url,
                api_key=client.api_key,
                local_track=video_source.track,
                options=RealtimeConnectOptions(
                    model=model,
                    on_remote_stream=on_remote_stream,
                    initial_state=ModelState(
                        prompt=Prompt(
                            text="use the image as a reference",
                            enhance=True,
                        ),
                        image=Path("examples/files/image.png"),
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
                try:
                    await realtime_client.set_prompt("Cyberpunk city")
                    print("✓ Prompt set successfully")
                except Exception as e:
                    print(f"⚠️ Failed to set prompt: {e}")

                await asyncio.sleep(5)

                print(f"\n✓ Processed {frame_count} frames total")
            finally:
                print(f"Remote frame count written to console; placeholder output: {output_file}")

        except Exception as e:
            print(f"\n❌ Connection failed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            if "realtime_client" in locals():
                print("\nDisconnecting...")
                await realtime_client.disconnect()
                print("✓ Disconnected")
            video_source.stop()
            source_task.cancel()
            try:
                await source_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    asyncio.run(main())
