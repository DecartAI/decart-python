import asyncio
import os
from pathlib import Path
import cv2
from decart import DecartClient, models

try:
    from livekit import rtc
except ImportError:
    print("livekit is required for this example.")
    print("Install with: pip install decart[realtime]")
    exit(1)


class FileVideoSource:
    """Reads frames from a video file and publishes them through a LiveKit source."""

    def __init__(self, path: str, width: int, height: int, fps: int):
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.source = rtc.VideoSource(width, height)
        self.track = rtc.LocalVideoTrack.create_video_track("file-video", self.source)
        self._running = False

    async def start(self):
        capture = cv2.VideoCapture(self.path)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video file: {self.path}")

        self._running = True
        frame_interval = 1 / self.fps
        try:
            while self._running:
                ok, frame = capture.read()
                if not ok:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame = cv2.resize(frame, (self.width, self.height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.source.capture_frame(
                    rtc.VideoFrame(
                        width=self.width,
                        height=self.height,
                        type=rtc.VideoBufferType.RGB24,
                        data=frame.tobytes(),
                    )
                )
                await asyncio.sleep(frame_interval)
        finally:
            capture.release()

    def stop(self):
        self._running = False


async def main():
    api_key = os.getenv("DECART_API_KEY")
    if not api_key:
        print("Error: DECART_API_KEY environment variable not set")
        print("Usage: DECART_API_KEY=your-key python realtime_file.py <video_file>")
        return

    import sys

    if len(sys.argv) < 2:
        print("Usage: python realtime_file.py <video_file>")
        print("Example: python realtime_file.py output_t2v.mp4")
        return

    video_file = sys.argv[1]
    if not os.path.exists(video_file):
        print(f"Error: Video file not found: {video_file}")
        return

    try:
        from decart.realtime.client import RealtimeClient
    except ImportError:
        print("Error: Realtime API not available")
        print("Install with: pip install decart[realtime]")
        return

    print("Creating Decart client...")
    async with DecartClient(api_key=api_key) as client:
        model = models.realtime("lucy-restyle-2")
        print(f"Using model: {model.name}")

        frame_count = 0
        input_name = Path(video_file).stem
        output_file = Path(f"output_realtime_{input_name}.frames")
        video_source = FileVideoSource(video_file, model.width, model.height, model.fps)
        source_task = asyncio.create_task(video_source.start())

        def on_remote_stream(track):
            print(f"📹 Received remote LiveKit track: {track.sid}")

            async def consume_frames():
                nonlocal frame_count
                async for event in rtc.VideoStream(track):
                    frame_count += 1
                    if frame_count % 25 == 0:
                        print(f"📹 Processed {frame_count} remote frames...")

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
                    initial_state=ModelState(prompt=Prompt(text="Lego World", enhance=True)),
                ),
            )

            realtime_client.on("connection_change", on_connection_change)
            realtime_client.on("error", on_error)

            print("✓ Connected!")
            print(f"Session ID: {realtime_client.session_id}")
            print("Processing video... (Ctrl+C to stop)")

            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print(f"\n\n✓ Processed {frame_count} frames total")
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
