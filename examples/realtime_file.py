import asyncio
import os
from pathlib import Path
from decart import DecartClient, models

try:
    from aiortc.contrib.media import MediaPlayer, MediaRecorder
except ImportError:
    print("aiortc is required for this example.")
    print("Install with: pip install decart[realtime]")
    exit(1)


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

    print(f"Loading video file: {video_file}")
    player = MediaPlayer(video_file)

    if not player.video:
        print("Error: No video stream found in file")
        return

    try:
        from decart.realtime.client import RealtimeClient
    except ImportError:
        print("Error: Realtime API not available")
        print("Install with: pip install decart[realtime]")
        return

    print("Creating Decart client...")
    async with DecartClient(api_key=api_key) as client:
        model = models.realtime("mirage")
        print(f"Using model: {model.name}")

        frame_count = 0
        recorder = None
        input_name = Path(video_file).stem
        output_file = Path(f"output_realtime_{input_name}.mp4")

        def on_remote_stream(track):
            nonlocal frame_count, recorder
            frame_count += 1
            if frame_count % 25 == 0:
                print(f"📹 Processed {frame_count} frames...")

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
                local_track=player.video,
                options=RealtimeConnectOptions(
                    model=model,
                    on_remote_stream=on_remote_stream,
                    initial_state=ModelState(
                        prompt=Prompt(text="Lego World", enrich=True), mirror=False
                    ),
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
