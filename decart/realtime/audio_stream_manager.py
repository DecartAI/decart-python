"""
Audio stream manager for live_avatar mode.

Mirrors the JS SDK's AudioStreamManager — ensures WebRTC always has
audio frames to send even when no user mic/audio is provided.
"""

import asyncio
import fractions
import io
import logging
from collections import deque
from pathlib import Path
from typing import Optional, Union

import av
from aiortc import MediaStreamTrack

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
SAMPLES_PER_FRAME = 960  # 20ms at 48kHz
BYTES_PER_SAMPLE = 2  # s16 format
BYTES_PER_FRAME = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE


def _make_silence_frame() -> av.AudioFrame:
    frame = av.AudioFrame(samples=SAMPLES_PER_FRAME, layout="mono", format="s16")
    for plane in frame.planes:
        plane.update(bytes(BYTES_PER_FRAME))
    return frame


class _AudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._queue: deque[av.AudioFrame] = deque()
        self._pts = 0
        self._start: Optional[float] = None
        self._done_event: Optional[asyncio.Event] = None

    async def recv(self) -> av.AudioFrame:
        if self._start is None:
            self._start = asyncio.get_event_loop().time()

        target = self._start + (self._pts / SAMPLE_RATE)
        delay = target - asyncio.get_event_loop().time()
        if delay > 0:
            await asyncio.sleep(delay)

        if self._queue:
            frame = self._queue.popleft()
            if not self._queue and self._done_event:
                self._done_event.set()
                self._done_event = None
        else:
            frame = _make_silence_frame()

        frame.pts = self._pts
        frame.sample_rate = SAMPLE_RATE
        frame.time_base = fractions.Fraction(1, SAMPLE_RATE)
        self._pts += SAMPLES_PER_FRAME

        return frame

    def enqueue(self, frames: list[av.AudioFrame], done: asyncio.Event) -> None:
        self._queue.extend(frames)
        self._done_event = done

    def clear(self) -> None:
        self._queue.clear()
        if self._done_event:
            self._done_event.set()
            self._done_event = None


class AudioStreamManager:
    """Manages audio for live_avatar mode.

    Provides a continuous audio track that outputs silence by default
    and allows playing audio data through it via play_audio().
    """

    def __init__(self) -> None:
        self._track = _AudioTrack()
        self._playing = False

    def get_track(self) -> MediaStreamTrack:
        return self._track

    @property
    def is_playing(self) -> bool:
        return self._playing

    async def play_audio(self, audio: Union[bytes, str, Path]) -> None:
        """Play audio through the stream. Resolves when audio finishes playing.

        Args:
            audio: Audio data as bytes, file path string, or Path object.
        """
        if self._playing:
            self.stop_audio()

        if isinstance(audio, bytes):
            container: av.InputContainer = av.open(io.BytesIO(audio))  # type: ignore[assignment]
        else:
            container: av.InputContainer = av.open(str(audio))  # type: ignore[assignment]

        try:
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            raw = bytearray()

            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    raw.extend(bytes(resampled.planes[0]))

            for resampled in resampler.resample(None):
                raw.extend(bytes(resampled.planes[0]))
        finally:
            container.close()

        if not raw:
            return

        frames = []
        for i in range(0, len(raw), BYTES_PER_FRAME):
            chunk = raw[i : i + BYTES_PER_FRAME]
            if len(chunk) < BYTES_PER_FRAME:
                chunk.extend(bytes(BYTES_PER_FRAME - len(chunk)))

            frame = av.AudioFrame(samples=SAMPLES_PER_FRAME, layout="mono", format="s16")
            frame.planes[0].update(bytes(chunk))
            frames.append(frame)

        done = asyncio.Event()
        self._playing = True
        self._track.enqueue(frames, done)

        await done.wait()
        self._playing = False

    def stop_audio(self) -> None:
        self._track.clear()
        self._playing = False

    def cleanup(self) -> None:
        self.stop_audio()
        self._track.stop()
