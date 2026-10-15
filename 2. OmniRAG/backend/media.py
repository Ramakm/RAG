"""Audio & video: speech-to-text with faster-whisper and frame sampling with PyAV."""
from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path
from typing import Callable

_lock = threading.Lock()


@lru_cache(maxsize=2)
def _whisper(size: str):
    from faster_whisper import WhisperModel
    return WhisperModel(size, device="cpu", compute_type="int8")


def probe(path: Path) -> dict:
    """Duration and which streams a media file has."""
    import av

    with av.open(str(path)) as c:
        duration = float(c.duration / av.time_base) if c.duration else 0.0
        if not duration and c.streams.video:
            vs = c.streams.video[0]
            duration = float(vs.duration * vs.time_base) if vs.duration else 0.0
        return {"duration": round(duration, 2), "has_audio": bool(c.streams.audio), "has_video": bool(c.streams.video)}


def transcribe(path: Path, size: str, on_progress: Callable[[float], None]) -> tuple[list[dict], str | None]:
    """Returns ([{start, end, text}], language). Silence is skipped by the VAD filter."""
    with _lock:
        model = _whisper(size)
    segments, info = model.transcribe(str(path), beam_size=2, vad_filter=True,
                                      condition_on_previous_text=False)
    out = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            out.append({"start": float(seg.start), "end": float(seg.end), "text": text})
        if info.duration:
            on_progress(min(1.0, seg.end / info.duration))
    return out, info.language


def sample_frames(path: Path, assets: Path, fid: str, interval: float, max_frames: int, duration: float) -> list[dict]:
    """One frame every `interval` seconds (spread wider for long videos). Near-duplicates are
    removed later, after embedding, so static shots don't flood the index."""
    import av

    if duration <= 0:
        return []
    interval = max(interval, duration / max_frames)
    times = [t for t in (interval / 2 + i * interval for i in range(max_frames)) if t < duration]
    frames = []
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for t in times:
            try:
                c.seek(int(t * av.time_base), any_frame=False, backward=True)
                frame = None
                for n, f in enumerate(c.decode(stream)):
                    frame = f
                    if f.time is None or f.time >= t - 0.05 or n > 300:
                        break
                if frame is None:
                    continue
                img = frame.to_image()
                img.thumbnail((640, 640))
                dest = assets / f"{fid}-t{int(t)}.jpg"
                img.save(dest, "JPEG", quality=82)
                frames.append({"path": dest, "t": round(t, 2)})
            except Exception:
                continue
    return frames
