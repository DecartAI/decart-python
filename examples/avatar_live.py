"""
Avatar Live Example

This example demonstrates how to use the avatar-live model to animate an avatar image.
The avatar can be animated with audio input (microphone or audio file).

Usage:
    # With audio file:
    DECART_API_KEY=your-key python avatar_live.py avatar.png audio.mp3

    # With just avatar image (will wait for audio):
    DECART_API_KEY=your-key python avatar_live.py avatar.png

Requirements:
    pip install decart[realtime]
"""

import asyncio
import os
import sys
from pathlib import Path

try:
    from aiortc.contrib.media import MediaPlayer, MediaRecorder
except ImportError:
    print("aiortc is required for this example.")
    print("Install with: pip install decart[realtime]")
    sys.exit(1)

from decart import DecartClient, models


async def main():
    api_key = os.getenv("DECART_API_KEY")
    if not api_key:
        print("Error: DECART_API_KEY environment variable not set")
        print("Usage: DECART_API_KEY=your-key python avatar_live.py <avatar_image> [audio_file]")
        return

    if len(sys.argv) < 2:
        print("Usage: python avatar_live.py <avatar_image> [audio_file]")
        print("")
        print("Arguments:")
        print("  avatar_image  - Path to avatar image (PNG, JPG)")
        print("  audio_file    - Optional: Path to audio file for the avatar to speak")
        print("")
        print("Examples:")
        print("  python avatar_live.py avatar.png")
        print("  python avatar_live.py avatar.png speech.mp3")
        return

    avatar_image = sys.argv[1]
    if not os.path.exists(avatar_image):
        print(f"Error: Avatar image not found: {avatar_image}")
        return

    audio_file = sys.argv[2] if len(sys.argv) > 2 else None
    if audio_file and not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        return

    print(f"🖼️  Avatar image: {avatar_image}")
    if audio_file:
        print(f"🔊 Audio file: {audio_file}")

    # Load audio if provided
    audio_track = None
    if audio_file:
        print("Loading audio file...")
        player = MediaPlayer(audio_file)
        if player.audio:
            audio_track = player.audio
            print("✓ Audio loaded")
        else:
            print("⚠️  Warning: No audio stream found in file, continuing without audio")

    try:
        from decart.realtime.client import RealtimeClient
        from decart.realtime.types import RealtimeConnectOptions, AvatarOptions
    except ImportError:
        print("Error: Realtime API not available")
        print("Install with: pip install decart[realtime]")
        return

    print("\nCreating Decart client...")
    async with DecartClient(api_key=api_key) as client:
        model = models.realtime("avatar-live")
        print(f"Using model: {model.name}")

        frame_count = 0
        recorder = None
        output_file = Path(f"output_avatar_live.mp4")

        def on_remote_stream(track):
            nonlocal frame_count, recorder
            frame_count += 1
            if frame_count % 25 == 0:
                print(f"📹 Received {frame_count} frames...")

            if recorder is None:
                print(f"💾 Recording to {output_file}")
                recorder = MediaRecorder(str(output_file))
                recorder.addTrack(track)
                asyncio.create_task(recorder.start())

        def on_connection_change(state):
            print(f"🔄 Connection state: {state}")

        def on_error(error):
            print(f"❌ Error: {error.__class__.__name__} - {error.message}")

        print("\nConnecting to Avatar Live API...")
        print("(Sending avatar image...)")

        try:
            realtime_client = await RealtimeClient.connect(
                base_url=client.base_url,
                api_key=client.api_key,
                local_track=audio_track,  # Can be None if no audio
                options=RealtimeConnectOptions(
                    model=model,
                    on_remote_stream=on_remote_stream,
                    avatar=AvatarOptions(avatar_image=Path(avatar_image)),
                ),
            )

            realtime_client.on("connection_change", on_connection_change)
            realtime_client.on("error", on_error)

            print("✓ Connected!")
            print(f"Session ID: {realtime_client.session_id}")

            if audio_file:
                print("\nPlaying audio through avatar...")
                print("(The avatar will animate based on the audio)")
            else:
                print("\nNo audio provided - avatar will be static")
                print("You can update the avatar image dynamically using set_image()")

            print("\nPress Ctrl+C to stop and save the recording...")

            # Demo: Update avatar image after 5 seconds (if you want to test set_image)
            # Uncomment the following to test dynamic image updates:
            # await asyncio.sleep(5)
            # print("Updating avatar image...")
            # await realtime_client.set_image(Path("new_avatar.png"))
            # print("✓ Avatar image updated!")

            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print(f"\n\n✓ Received {frame_count} frames total")

            finally:
                if recorder:
                    try:
                        print(f"💾 Saving video to {output_file}...")
                        await asyncio.sleep(0.5)
                        await recorder.stop()
                        print(f"✓ Video saved to {output_file}")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not save video cleanly: {e}")
                        print("   Video file may be incomplete")

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
