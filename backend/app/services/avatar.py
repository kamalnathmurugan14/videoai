"""Avatar animation. Providers produce one RGBA frame per video frame.

AmplitudeAvatarProvider (default): audio loudness drives mouth opening, plus blink + subtle head bob.
CommandAvatarProvider: shells out to an external lip-sync model (SadTalker / Wav2Lip ...) configured via
AVATAR_PROVIDER=command and AVATAR_COMMAND. Add your own by subclassing AvatarProvider.
"""
import math
import shlex
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw

from ..config import get_settings
from .tts import wav_samples

DEFAULT_META = {"mouth": [0.5, 0.67, 0.2], "eyes": None, "skin": "#e0ac8a", "has_mouth_art": False}


def amplitude_envelope(wav: Path, fps: int, n_frames: int) -> np.ndarray:
    """Per-video-frame mouth openness in [0,1], smoothed."""
    data, sr = wav_samples(wav)
    hop = sr / fps
    env = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = data[int(i * hop): int((i + 1) * hop) + int(hop)]
        env[i] = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
    peak = np.percentile(env[env > 0], 95) if np.any(env > 0) else 1.0
    env = np.clip(env / max(peak, 1e-6), 0, 1) ** 0.8
    env[env < 0.06] = 0
    out = env.copy()
    for i in range(1, n_frames):  # fast attack, slower release
        out[i] = max(env[i], out[i - 1] * 0.55) if env[i] < out[i - 1] else 0.5 * out[i - 1] + 0.5 * env[i]
    return out


def speech_bounds(env: np.ndarray, fps: int) -> tuple[float, float]:
    idx = np.where(env > 0)[0]
    if len(idx) == 0:
        return 0.0, len(env) / fps
    return idx[0] / fps, (idx[-1] + 1) / fps


def draw_mouth(img: Image.Image, cx: float, cy: float, width: float, openness: float, lip="#a8324a") -> None:
    """Draw an anti-aliased mouth onto `img` (RGBA) centred at (cx, cy) with the given width."""
    ss = 4
    pw, ph = int(width * 1.6), int(width * 1.2)
    patch = Image.new("RGBA", (pw * ss, ph * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(patch)
    mx, my = pw * ss / 2, ph * ss / 2
    w = width * ss / 2
    open_h = openness * width * 0.55 * ss
    lw = max(2, int(width * 0.06 * ss))
    if open_h < width * 0.04 * ss:  # closed: gentle smile
        d.arc([mx - w, my - w * 0.5, mx + w, my + w * 0.5], 20, 160, fill=lip, width=lw)
    else:
        top = my - open_h * 0.25
        bottom = my + open_h
        d.pieslice([mx - w, top - w * 0.3, mx + w, bottom + w * 0.1], 0, 180, fill="#3b0d1a")  # lower mouth bowl
        d.rectangle([mx - w, top - w * 0.3, mx + w, top + w * 0.02], fill="#3b0d1a")
        d.rectangle([mx - w * 0.7, top - w * 0.02, mx + w * 0.7, top + open_h * 0.22], fill="#f8f4f0")  # teeth
        d.ellipse([mx - w * 0.5, bottom - open_h * 0.45, mx + w * 0.5, bottom + open_h * 0.1], fill="#e0707f")  # tongue
        d.arc([mx - w, top - w * 0.3, mx + w, bottom + w * 0.1], 0, 180, fill=lip, width=lw)
        d.line([mx - w, top - w * 0.15, mx + w, top - w * 0.15], fill=lip, width=lw)
    patch = patch.resize((pw, ph), Image.LANCZOS)
    img.alpha_composite(patch, (int(cx - pw / 2), int(cy - ph / 2)))


def draw_blink(img: Image.Image, eyes, skin: str, size: int) -> None:
    d = ImageDraw.Draw(img)
    for ex, ey, er in eyes:
        x, y, r = ex * size, ey * size, er * size
        d.ellipse([x - r * 1.15, y - r * 0.8, x + r * 1.15, y + r * 0.8], fill=skin)
        d.arc([x - r, y - r * 0.5, x + r, y + r * 0.5], 10, 170, fill="#2b1b17", width=max(2, int(r * 0.25)))


class AvatarProvider(ABC):
    name = "base"

    @abstractmethod
    def frames(self, image_path: Path, meta: dict, wav: Path, size: int, fps: int, n_frames: int, workdir: Path) -> Iterator[Image.Image]:
        """Yield `n_frames` RGBA images of shape (size, size)."""


class AmplitudeAvatarProvider(AvatarProvider):
    name = "amplitude"

    def frames(self, image_path, meta, wav, size, fps, n_frames, workdir):
        base = Image.open(image_path).convert("RGBA")
        side = max(base.size)  # pad to square so mouth fractions are consistent
        sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        sq.paste(base, ((side - base.width) // 2, (side - base.height) // 2))
        base = sq.resize((size, size), Image.LANCZOS)
        meta = {**DEFAULT_META, **(meta or {})}
        mx, my, mw = meta["mouth"]
        env = amplitude_envelope(wav, fps, n_frames)
        for i in range(n_frames):
            t = i / fps
            a = float(env[i])
            frame = base.copy()
            blink = (t % 3.6) < 0.14 and t > 0.5
            if blink and meta["eyes"]:
                draw_blink(frame, meta["eyes"], meta["skin"], size)
            if meta["has_mouth_art"]:
                draw_mouth(frame, mx * size, my * size, mw * size, a)
            else:  # arbitrary uploaded photo: stretch the lower face (jaw drop) and add a dark mouth opening
                d = int(a * 0.035 * size)
                if d > 1:
                    y0 = int(my * size)
                    y1 = min(size, y0 + int(0.22 * size))
                    strip = base.crop((0, y0, size, y1)).resize((size, y1 - y0 + d), Image.BICUBIC)
                    frame.paste(strip, (0, y0))
                    gap = ImageDraw.Draw(frame)
                    gw, gh = mw * size / 2, d * 0.6
                    gap.ellipse([mx * size - gw, y0 - gh * 0.3, mx * size + gw, y0 + gh * 1.2], fill=(40, 10, 20, 235))
            # head bob + talk bounce
            dy = int(math.sin(t * 2.1) * size * 0.004 + a * size * 0.004)
            if dy:
                shifted = Image.new("RGBA", frame.size, (0, 0, 0, 0))
                shifted.paste(frame, (0, dy))
                frame = shifted
            yield frame


class CommandAvatarProvider(AvatarProvider):
    """Runs AVATAR_COMMAND ({image} {audio} {out} placeholders) to produce a talking-head video, then reads frames.
    Output is composited as an opaque square (most lip-sync models do not output alpha)."""

    name = "command"

    def frames(self, image_path, meta, wav, size, fps, n_frames, workdir):
        cmd = get_settings().avatar_command
        if not cmd:
            raise RuntimeError("AVATAR_COMMAND is not set")
        out = workdir / "lipsync.mp4"
        parts = [p.format(image=str(image_path), audio=str(wav), out=str(out)) for p in shlex.split(cmd)]
        r = subprocess.run(parts, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"lip-sync command failed: {r.stderr[-300:]}")
        proc = subprocess.Popen(
            [get_settings().ffmpeg_bin, "-loglevel", "error", "-i", str(out), "-vf",
             f"fps={fps},scale={size}:{size}:force_original_aspect_ratio=decrease,pad={size}:{size}:(ow-iw)/2:(oh-ih)/2",
             "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
            stdout=subprocess.PIPE,
        )
        n = size * size * 4
        last = None
        for _ in range(n_frames):
            buf = proc.stdout.read(n)
            if len(buf) == n:
                last = Image.frombytes("RGBA", (size, size), buf)
            if last is None:
                last = Image.open(image_path).convert("RGBA").resize((size, size))
            yield last
        proc.stdout.close()
        proc.wait()


def get_provider() -> AvatarProvider:
    return CommandAvatarProvider() if get_settings().avatar_provider == "command" else AmplitudeAvatarProvider()
