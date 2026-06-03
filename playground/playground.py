#!/usr/bin/env python3
"""
Decart Python SDK — Local LiveKit Playground

OpenCV-based CLI playground for testing the Decart realtime API.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional, cast


def _check_deps() -> None:
    missing: list[str] = []
    try:
        import cv2  # noqa: F401
    except ImportError:
        missing.append("opencv-python")
    try:
        from livekit import rtc  # noqa: F401
    except ImportError:
        missing.append("decart[realtime]  (includes livekit)")
    try:
        from decart import DecartClient  # noqa: F401
    except ImportError:
        missing.append("decart")
    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with:")
        print("  pip install opencv-python")
        print("  pip install -e ..[realtime]   # local dev")
        print("  # or: pip install decart[realtime]")
        sys.exit(1)


_check_deps()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from livekit import rtc  # noqa: E402

from decart import DecartClient, models  # noqa: E402
from decart.models import RealTimeModels  # noqa: E402
from decart.realtime.client import RealtimeClient  # noqa: E402
from decart.realtime.types import RealtimeConnectOptions  # noqa: E402
from decart.types import ModelState, Prompt  # noqa: E402

logging.basicConfig(level=logging.WARNING)

REALTIME_MODELS = [
    "lucy-2.1",
    "lucy-2.1-vton",
    "lucy-restyle-2",
]

BANNER = """
╔══════════════════════════════════════╗
║   Decart Python SDK — Playground    ║
╚══════════════════════════════════════╝"""


class CameraVideoSource:
    def __init__(self, device: int, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {device}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self.source = rtc.VideoSource(width, height)
        self.track = rtc.LocalVideoTrack.create_video_track("camera-video", self.source)
        self.last_frame: Optional[np.ndarray] = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        frame_interval = 1 / self.fps
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                await asyncio.sleep(frame_interval)
                continue
            frame = cv2.resize(frame, (self.width, self.height))
            self.last_frame = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.source.capture_frame(
                rtc.VideoFrame(
                    width=self.width,
                    height=self.height,
                    type=rtc.VideoBufferType.RGB24,
                    data=rgb.tobytes(),
                )
            )
            await asyncio.sleep(frame_interval)

    def stop(self) -> None:
        self._running = False
        if self._cap.isOpened():
            self._cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decart Python SDK LiveKit playground")
    parser.add_argument("--model", "-m", choices=REALTIME_MODELS, help="Model name")
    parser.add_argument("--api-key", "-k", help="API key (or set DECART_API_KEY)")
    parser.add_argument("--image", "-i", help="Optional reference image")
    parser.add_argument("--prompt", "-p", help="Initial prompt text")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device index")
    parser.add_argument("--no-local", action="store_true", help="Hide local camera feed")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def select_model_interactive() -> str:
    print("\nAvailable realtime models:")
    for i, name in enumerate(REALTIME_MODELS, 1):
        print(f"  {i}. {name}")

    while True:
        choice = input(f"\nSelect model [1-{len(REALTIME_MODELS)}]: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(REALTIME_MODELS):
                return REALTIME_MODELS[idx]
        except ValueError:
            if choice in REALTIME_MODELS:
                return choice
        print("  Invalid choice, try again")


async def run() -> None:
    args = parse_args()
    print(BANNER)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    api_key = args.api_key or os.getenv("DECART_API_KEY") or input("Enter your Decart API key: ")
    if not api_key:
        print("Error: API key is required")
        return

    model_name = args.model or select_model_interactive()
    model = models.realtime(cast(RealTimeModels, model_name))
    print(f"\n  Model : {model_name}")
    print(f"  Res   : {model.width}x{model.height} @ {model.fps}fps")

    if args.image and not Path(args.image).exists():
        print(f"\nError: Image not found: {args.image}")
        return

    initial_state: Optional[ModelState] = None
    if args.image or args.prompt:
        initial_state = ModelState(
            image=args.image if args.image else None,
            prompt=Prompt(text=args.prompt) if args.prompt else None,
        )

    camera = CameraVideoSource(args.camera, model.width, model.height, model.fps)
    camera_task = asyncio.create_task(camera.start())
    remote_track_ready = asyncio.Event()
    latest_remote: list[Optional[np.ndarray]] = [None]

    def on_remote_stream(track) -> None:
        print(f"  ✓ Remote LiveKit track received: {track.sid}")

        async def consume_frames() -> None:
            async for event in rtc.VideoStream(track):
                frame = event.frame.convert(rtc.VideoBufferType.RGB24)
                data = np.frombuffer(frame.data, dtype=np.uint8).reshape(
                    (frame.height, frame.width, 3)
                )
                latest_remote[0] = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
                remote_track_ready.set()

        asyncio.create_task(consume_frames())

    def on_connection_change(state: str) -> None:
        print(f"  Connection: {state}")

    def on_error(error: Exception) -> None:
        print(f"  Error: {error}")

    prompt_queue: asyncio.Queue[str] = asyncio.Queue()
    stop = threading.Event()
    loop = asyncio.get_running_loop()

    def _read_prompts() -> None:
        while not stop.is_set():
            try:
                line = input("prompt> ").strip()
                if line:
                    asyncio.run_coroutine_threadsafe(prompt_queue.put(line), loop)
            except (EOFError, KeyboardInterrupt):
                asyncio.run_coroutine_threadsafe(prompt_queue.put("/quit"), loop)
                break

    reader = threading.Thread(target=_read_prompts, daemon=True)
    reader.start()

    realtime: Optional[RealtimeClient] = None
    try:
        client = DecartClient(api_key=api_key)
        realtime = await RealtimeClient.connect(
            base_url=client.realtime_base_url,
            api_key=client.api_key,
            local_track=camera.track,
            options=RealtimeConnectOptions(
                model=model,
                on_remote_stream=on_remote_stream,
                initial_state=initial_state,
            ),
        )
        realtime.on("connection_change", on_connection_change)
        realtime.on("error", on_error)
        print(f"  ✓ Connected! Session: {realtime.session_id}")

        while True:
            while not prompt_queue.empty():
                line = await prompt_queue.get()
                if line in ("/quit", "q"):
                    return
                if line.startswith("/image "):
                    await realtime.set_image(line[len("/image ") :].strip())
                else:
                    await realtime.set_prompt(
                        line[5:].strip() if line.startswith("/set ") else line
                    )

            if not args.no_local and camera.last_frame is not None:
                cv2.imshow("Local Camera", camera.last_frame)
            if latest_remote[0] is not None:
                cv2.imshow("Decart Remote", latest_remote[0])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

            await asyncio.sleep(0.01)
    finally:
        stop.set()
        camera.stop()
        camera_task.cancel()
        try:
            await camera_task
        except asyncio.CancelledError:
            pass
        if realtime:
            await realtime.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(run())
