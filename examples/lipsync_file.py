#!/usr/bin/env python
"""
Example of using Decart's Realtime Lipsync API to synchronize audio with video.

This example loads a video file and an audio file, processes them through the
Decart Lipsync API, and saves the lipsynced result to a new video file.

Usage:
    python lipsync_file.py <video_file> <audio_file> <output_file>

Example:
    python lipsync_file.py input.mp4 speech.wav output_lipsynced.mp4
    python lipsync_file.py input.mp4 speech.mp3 output_lipsynced.mp4
"""

import asyncio
import os
import sys
import cv2
from pathlib import Path

from decart.lipsync import RealtimeLipsyncClient


async def process_lipsync(video_path: str, audio_path: str, output_path: str):
    """Process video and audio through Decart's lipsync API."""

    # Get API key
    api_key = os.getenv("DECART_API_KEY")
    if not api_key:
        print("Error: Please set DECART_API_KEY environment variable")
        return

    # Initialize client
    client = RealtimeLipsyncClient(api_key=api_key)

    print(f"Processing: {video_path} + {audio_path} -> {output_path}")

    # Connect to server
    await client.connect()
    print("Connected to Decart Lipsync server")

    try:
        # Load audio data - handle different formats
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        # Send audio to server (server handles chunking)
        await client.send_audio(audio_data)

        # Load video frames and convert to RGB
        frame_count = 0
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Convert from BGR (OpenCV default) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(rgb_frame.shape)
            await client.send_video_frame(rgb_frame)
        cap.release()

        # Receive lipsynced output
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            client._video_fps,
            (rgb_frame.shape[1], rgb_frame.shape[0]),
        )
        for i in range(frame_count):
            try:
                video_frame, audio_frame = await client.get_synced_output(timeout=1.0)
                bgr_frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
            except asyncio.TimeoutError:
                print(f"Warning: Timeout at frame {i}")
                break
        out.release()

    finally:
        await client.disconnect()
        print("Disconnected from server")


async def main():
    """Main entry point."""
    if len(sys.argv) != 4:
        print("Usage: python lipsync_file.py <video_file> <wav_audio_file> <output_file>")
        print("Example: python lipsync_file.py input.mp4 speech.wav output_lipsynced.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    audio_path = sys.argv[2]
    output_path = sys.argv[3]

    # Check input files exist
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    # Process the files
    await process_lipsync(video_path, audio_path, output_path)


if __name__ == "__main__":
    asyncio.run(main())
