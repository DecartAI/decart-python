#!/usr/bin/env python3
"""
Decart Python SDK — Local Playground

OpenCV-based CLI playground for testing the Decart realtime API.

Features:
  1. Model selection (CLI arg or interactive prompt)
  2. API key input (env var, CLI arg, or interactive prompt)
  3. Live camera → Decart → display window
  4. Prompt input (type in terminal, sent to Decart)
  5. Image reference (file path arg for initial state)

Usage:
  python playground.py                                      # Interactive mode
  python playground.py --model mirage_v2                    # Camera model
  python playground.py --model mirage_v2 --prompt "Anime"   # With initial prompt
  python playground.py --model avatar-live --image face.png  # Avatar mode
  python playground.py --model avatar-live --image face.png --audio speech.mp3

Controls (while running):
  Type text + Enter    → Send prompt to Decart
  /image <path>        → Send reference image
  /set <prompt>        → Send prompt + enhance=True (same as plain text)
  /quit or 'q' key     → Exit

Requirements:
  pip install opencv-python
  pip install decart[realtime]   # or: pip install -e ..[realtime]
"""

from __future__ import annotations

import argparse
import asyncio
import fractions
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, cast

# ── Dependency checks ────────────────────────────────────────────────────────


def _check_deps() -> None:
    missing: list[str] = []
    try:
        import cv2  # noqa: F401
    except ImportError:
        missing.append("opencv-python")
    try:
        import av  # noqa: F401
        from aiortc import MediaStreamTrack  # noqa: F401
    except ImportError:
        missing.append("decart[realtime]  (includes aiortc)")
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
from av import VideoFrame  # noqa: E402
from aiortc import MediaStreamTrack  # noqa: E402
from aiortc.mediastreams import MediaStreamError, VideoStreamTrack  # noqa: E402

from decart import DecartClient, models  # noqa: E402
from decart.models import RealTimeModels  # noqa: E402
from decart.realtime.client import RealtimeClient  # noqa: E402
from decart.realtime.types import RealtimeConnectOptions  # noqa: E402
from decart.types import ModelState, Prompt  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("playground")


# ── Constants ────────────────────────────────────────────────────────────────

REALTIME_MODELS = ["mirage", "mirage_v2", "lucy_v2v_720p_rt", "lucy_2_rt", "avatar-live"]
CAMERA_MODELS = {"mirage", "mirage_v2", "lucy_v2v_720p_rt", "lucy_2_rt"}
AVATAR_MODELS = {"avatar-live"}

BANNER = """
╔══════════════════════════════════════╗
║   Decart Python SDK — Playground    ║
╚══════════════════════════════════════╝"""


# ── Camera Track ─────────────────────────────────────────────────────────────


class CameraTrack(VideoStreamTrack):
    """Bridges OpenCV webcam capture to an aiortc video track."""

    kind = "video"

    def __init__(self, device: int, width: int, height: int, fps: int) -> None:
        super().__init__()
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {device}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._fps = fps
        self._count = 0
        self._t0: Optional[float] = None
        self.last_frame: Optional[np.ndarray] = None

    async def recv(self) -> VideoFrame:
        if self._t0 is None:
            self._t0 = time.time()

        # Pace output to target FPS
        target = self._t0 + self._count / self._fps
        delay = target - time.time()
        if delay > 0:
            await asyncio.sleep(delay)

        ret, frame = self._cap.read()
        if not ret:
            raise MediaStreamError("Camera read failed")

        self.last_frame = frame.copy()

        vf = VideoFrame.from_ndarray(frame, format="bgr24")  # type: ignore[arg-type]
        vf.pts = self._count
        vf.time_base = fractions.Fraction(1, self._fps)
        self._count += 1
        return vf

    def stop(self) -> None:
        super().stop()
        if self._cap.isOpened():
            self._cap.release()


# ── Audio Track ──────────────────────────────────────────────────────────────


def load_audio_track(path: str) -> Optional[MediaStreamTrack]:
    """Load an audio track from a file using aiortc's MediaPlayer."""
    try:
        from aiortc.contrib.media import MediaPlayer

        player = MediaPlayer(path)
        if player.audio:
            return player.audio
        print(f"  ⚠ No audio stream in {path}")
    except Exception as e:
        print(f"  ⚠ Failed to load audio: {e}")
    return None


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Decart Python SDK — Local Playground",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --model mirage_v2
  %(prog)s --model mirage_v2 --prompt "Anime style"
  %(prog)s --model avatar-live --image avatar.png
  %(prog)s --model avatar-live --image avatar.png --audio speech.mp3
  %(prog)s --model lucy_2_rt --image ref.png --prompt "Lego World"
""",
    )
    p.add_argument("--model", "-m", choices=REALTIME_MODELS, help="Model name")
    p.add_argument("--api-key", "-k", help="API key (or set DECART_API_KEY env var)")
    p.add_argument("--image", "-i", help="Initial image path (required for avatar-live)")
    p.add_argument("--prompt", "-p", help="Initial prompt text")
    p.add_argument("--audio", "-a", help="Audio file path (for avatar-live)")
    p.add_argument("--camera", "-c", type=int, default=0, help="Camera device index (default: 0)")
    p.add_argument("--no-local", action="store_true", help="Hide local camera feed")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return p.parse_args()


def select_model_interactive() -> str:
    """Interactive model selection menu."""
    print("\nAvailable realtime models:")
    for i, name in enumerate(REALTIME_MODELS, 1):
        note = ""
        if name in AVATAR_MODELS:
            note = " (requires --image)"
        elif name in ("lucy_2_rt", "mirage_v2"):
            note = " (supports reference image)"
        print(f"  {i}. {name}{note}")

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


# ── Main ─────────────────────────────────────────────────────────────────────


async def run() -> None:
    args = parse_args()
    print(BANNER)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── API Key ──────────────────────────────────────────────────────────
    api_key = args.api_key or os.getenv("DECART_API_KEY")
    if not api_key:
        api_key = input("Enter your Decart API key: ").strip()
    if not api_key:
        print("Error: API key is required")
        return

    # ── Model Selection ──────────────────────────────────────────────────
    model_name = args.model
    if not model_name:
        model_name = select_model_interactive()

    model = models.realtime(cast(RealTimeModels, model_name))
    needs_camera = model_name in CAMERA_MODELS
    is_avatar = model_name in AVATAR_MODELS

    print(f"\n  Model : {model_name}")
    print(f"  Res   : {model.width}x{model.height} @ {model.fps}fps")

    # ── Validate ─────────────────────────────────────────────────────────
    if is_avatar and not args.image:
        print("\nError: --image is required for avatar-live model")
        return

    if args.image and not Path(args.image).exists():
        print(f"\nError: Image not found: {args.image}")
        return

    if args.audio and not Path(args.audio).exists():
        print(f"\nError: Audio file not found: {args.audio}")
        return

    # ── Initial State ────────────────────────────────────────────────────
    initial_state: Optional[ModelState] = None
    if args.image or args.prompt:
        initial_state = ModelState(
            image=args.image if args.image else None,
            prompt=Prompt(text=args.prompt) if args.prompt else None,
        )
        parts: list[str] = []
        if args.image:
            parts.append(f"image={Path(args.image).name}")
        if args.prompt:
            parts.append(f'prompt="{args.prompt}"')
        print(f"  Init  : {', '.join(parts)}")

    # ── Local Track ──────────────────────────────────────────────────────
    camera_track: Optional[CameraTrack] = None
    local_track: Optional[MediaStreamTrack] = None

    if needs_camera:
        print(f"\n  Opening camera (device {args.camera})...")
        try:
            camera_track = CameraTrack(args.camera, model.width, model.height, model.fps)
            local_track = camera_track
            actual_w = int(camera_track._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(camera_track._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  ✓ Camera opened ({actual_w}x{actual_h})")
        except RuntimeError as e:
            print(f"  ✗ {e}")
            return
    elif args.audio:
        print(f"  Loading audio: {args.audio}")
        local_track = load_audio_track(args.audio)
        if local_track:
            print("  ✓ Audio loaded")

    # ── Connect ──────────────────────────────────────────────────────────
    remote_track_ready = asyncio.Event()
    remote_video_track: list[Optional[MediaStreamTrack]] = [None]

    def on_remote_stream(track: MediaStreamTrack) -> None:
        remote_video_track[0] = track
        # Schedule event set on the running loop (callback may fire from aiortc thread)
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(remote_track_ready.set)
        except RuntimeError:
            remote_track_ready.set()

    def on_connection_change(state: str) -> None:
        print(f"  🔄 Connection: {state}")

    def on_error(error: Exception) -> None:
        print(f"  ❌ Error: {error}")

    print("\n  Connecting to Decart...")

    realtime: Optional[RealtimeClient] = None
    try:
        client = DecartClient(api_key=api_key)

        realtime = await RealtimeClient.connect(
            base_url=client.base_url,
            api_key=client.api_key,
            local_track=local_track,
            options=RealtimeConnectOptions(
                model=model,
                on_remote_stream=on_remote_stream,
                initial_state=initial_state,
            ),
        )

        realtime.on("connection_change", on_connection_change)
        realtime.on("error", on_error)

        print(f"  ✓ Connected! Session: {realtime.session_id}")

        print("  Waiting for remote stream...")
        try:
            await asyncio.wait_for(remote_track_ready.wait(), timeout=15.0)
            print("  ✓ Remote stream received\n")
        except asyncio.TimeoutError:
            print("  ⚠ No remote stream (timeout) — display may not work\n")

        # ── Controls ─────────────────────────────────────────────────────
        print("  ┌─ Controls ─────────────────────────────")
        print("  │ Type text + Enter  → Send prompt")
        print("  │ /image <path>      → Send reference image")
        print("  │ /quit or 'q' key   → Exit")
        print("  └─────────────────────────────────────────")
        print("  (Click the terminal to type prompts)\n")

        # ── Prompt reader thread ─────────────────────────────────────────
        prompt_queue: asyncio.Queue[str] = asyncio.Queue()
        stop = threading.Event()
        loop = asyncio.get_running_loop()

        def _read_prompts() -> None:
            while not stop.is_set():
                try:
                    line = input("prompt> ")
                    if line.strip():
                        asyncio.run_coroutine_threadsafe(prompt_queue.put(line.strip()), loop)
                except (EOFError, KeyboardInterrupt):
                    asyncio.run_coroutine_threadsafe(prompt_queue.put("/quit"), loop)
                    break

        reader = threading.Thread(target=_read_prompts, daemon=True)
        reader.start()

        # ── Frame consumer ───────────────────────────────────────────────
        latest_remote: list[Optional[np.ndarray]] = [None]
        frame_count = 0
        fps_t0 = time.time()
        display_fps = 0.0

        async def _consume_frames() -> None:
            nonlocal frame_count, fps_t0, display_fps
            track = remote_video_track[0]
            if not track:
                return
            try:
                while True:
                    frame = await track.recv()
                    video_frame = cast("VideoFrame", frame)
                    latest_remote[0] = video_frame.to_ndarray(format="bgr24")
                    frame_count += 1
                    elapsed = time.time() - fps_t0
                    if elapsed >= 1.0:
                        display_fps = frame_count / elapsed
                        frame_count = 0
                        fps_t0 = time.time()
            except (MediaStreamError, asyncio.CancelledError):
                pass

        consumer = asyncio.create_task(_consume_frames())

        # ── Main display + command loop ──────────────────────────────────
        window_name = f"Decart Playground — {model_name}"
        show_local = not args.no_local and camera_track is not None
        window_created = False
        running = True

        try:
            while running:
                while not prompt_queue.empty():
                    try:
                        cmd = prompt_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    if cmd == "/quit":
                        running = False
                        break
                    elif cmd.startswith("/image "):
                        img_path = cmd[7:].strip()
                        if not Path(img_path).exists():
                            print(f"  File not found: {img_path}")
                            continue
                        try:
                            print(f"  Sending image: {img_path}")
                            await realtime.set_image(img_path)
                            print("  ✓ Image sent")
                        except Exception as e:
                            print(f"  ✗ Failed: {e}")
                    elif cmd.startswith("/"):
                        print(f"  Unknown command: {cmd}")
                        print("  Available: /image <path>, /quit")
                    else:
                        try:
                            await realtime.set_prompt(cmd)
                            print("  ✓ Prompt acknowledged")
                        except Exception as e:
                            print(f"  ✗ Prompt failed: {e}")

                if not running:
                    break

                remote = latest_remote[0]
                if remote is not None:
                    local_frame = camera_track.last_frame if camera_track else None
                    if show_local and local_frame is not None:
                        h, w = remote.shape[:2]
                        local_resized = cv2.resize(local_frame, (w, h))
                        display = np.hstack([local_resized, remote])
                    else:
                        display = remote

                    if display_fps > 0:
                        cv2.putText(
                            display,
                            f"{display_fps:.1f} fps",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )

                    cv2.imshow(window_name, display)
                    window_created = True

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    running = False
                    break

                # Check if window was closed by user
                if window_created:
                    try:
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            running = False
                            break
                    except cv2.error:
                        pass

                await asyncio.sleep(0.005)

        except KeyboardInterrupt:
            pass

        # ── Cleanup ──────────────────────────────────────────────────────
        print("\n  Shutting down...")
        stop.set()
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass
        await realtime.disconnect()
        if camera_track:
            camera_track.stop()
        cv2.destroyAllWindows()
        print("  ✓ Done")

    except KeyboardInterrupt:
        print("\n  Interrupted")
        if realtime:
            await realtime.disconnect()
        if camera_track:
            camera_track.stop()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"\n  ✗ Connection failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        if camera_track:
            camera_track.stop()
        cv2.destroyAllWindows()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
