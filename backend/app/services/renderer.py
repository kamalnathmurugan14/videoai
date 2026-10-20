"""Render pipeline: per scene TTS -> animated avatar frames -> captions -> FFmpeg; then concat + music."""
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ..config import get_settings
from . import avatar as avatar_mod
from . import captions as cap
from . import tts


class RenderError(RuntimeError):
    pass


@dataclass
class SceneSpec:
    text: str
    background: dict
    media_path: Path | None = None


@dataclass
class RenderSpec:
    scenes: list[SceneSpec]
    aspect_ratio: str = "16:9"
    avatar_path: Path | None = None
    avatar_meta: dict = field(default_factory=dict)
    voice: str = "en-US-AriaNeural"
    speed: float = 1.0
    captions_enabled: bool = True
    caption_style: str = "bold"
    music_path: Path | None = None
    music_volume: float = 0.15


@dataclass
class RenderResult:
    video: Path
    thumbnail: Path
    duration: float
    notes: list[str]


Progress = Callable[[int, str], None]
BASE_DIMS = {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (720, 720)}


def dims(aspect: str) -> tuple[int, int]:
    w, h = BASE_DIMS.get(aspect, BASE_DIMS["16:9"])
    sc = get_settings().render_scale
    return max(2, round(w * sc / 2) * 2), max(2, round(h * sc / 2) * 2)


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def make_background(bg: dict, w: int, h: int, image_path: Path | None) -> Image.Image:
    t = bg.get("type", "gradient")
    c1 = _hex(bg.get("color", "#4f46e5"))
    if t == "image" and image_path and image_path.exists():
        return ImageOps.fit(Image.open(image_path).convert("RGB"), (w, h), Image.LANCZOS)
    if t == "color":
        return Image.new("RGB", (w, h), c1)
    c2 = _hex(bg.get("color2", "#9333ea"))
    yy, xx = np.mgrid[0:h, 0:w]
    k = ((xx / w) * 0.45 + (yy / h) * 0.55)[..., None]
    arr = np.array(c1) * (1 - k) + np.array(c2) * k
    return Image.fromarray(arr.astype("uint8"), "RGB")


def layout(aspect: str, w: int, h: int, has_media: bool):
    """Returns (avatar_size, avatar_cx, avatar_bottom, media_box|None, caption_y)."""
    if aspect == "9:16":
        size = int(w * (0.82 if has_media else 1.2))
        media = (int(w * 0.07), int(h * 0.06), int(w * 0.93), int(h * 0.38)) if has_media else None
        return size, w // 2, h, media, int(h * (0.47 if has_media else 0.2))
    if aspect == "1:1":
        size = int(w * (0.6 if has_media else 0.9))
        media = (int(w * 0.06), int(h * 0.05), int(w * 0.94), int(h * 0.38)) if has_media else None
        return size, w // 2, h, media, int(h * (0.45 if has_media else 0.14))
    size = int(h * 0.92)
    cx = int(w * 0.74) if has_media else w // 2
    media = (int(w * 0.05), int(h * 0.12), int(w * 0.52), int(h * 0.74)) if has_media else None
    return size, cx, h, media, int(h * 0.80)


def _paint_media(base: Image.Image, path: Path, box) -> None:
    x0, y0, x1, y1 = box
    card = ImageOps.contain(Image.open(path).convert("RGB"), (x1 - x0, y1 - y0), Image.LANCZOS)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    px, py = cx - card.width // 2, cy - card.height // 2
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle([px + 6, py + 10, px + card.width + 6, py + card.height + 10], 18, fill=(0, 0, 0, 90))
    base.paste(Image.alpha_composite(base.convert("RGBA"), shadow).convert("RGB"))
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, card.width - 1, card.height - 1], 18, fill=255)
    base.paste(card, (px, py), mask)


def _run_ffmpeg(args: list[str], stdin_frames=None) -> None:
    cmd = [get_settings().ffmpeg_bin, "-y", "-loglevel", "error", *args]
    if stdin_frames is None:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RenderError(f"ffmpeg failed: {r.stderr[-400:]}")
        return
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in stdin_frames:
            proc.stdin.write(frame)
        proc.stdin.close()
        err = proc.stderr.read().decode()[-400:]
        if proc.wait() != 0:
            raise RenderError(f"ffmpeg failed: {err}")
    except BrokenPipeError:
        raise RenderError(f"ffmpeg exited early: {proc.stderr.read().decode()[-400:]}")
    except BaseException:
        proc.kill()
        raise


def probe_duration(path: Path) -> float:
    ffprobe = get_settings().ffmpeg_bin.replace("ffmpeg", "ffprobe")
    r = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def render_video(spec: RenderSpec, workdir: Path, out_video: Path, out_thumb: Path, progress: Progress,
                 bg_paths: dict[str, Path] | None = None) -> RenderResult:
    s = get_settings()
    fps = s.render_fps
    W, H = dims(spec.aspect_ratio)
    workdir.mkdir(parents=True, exist_ok=True)
    provider = avatar_mod.get_provider()
    notes: list[str] = []
    scenes = [sc for sc in spec.scenes if sc.text.strip()]
    if not scenes:
        raise RenderError("Add some script text to at least one scene")
    n = len(scenes)
    clips: list[Path] = []
    thumb_saved = False
    all_text = " ".join(sc.text for sc in scenes)
    cap_r = cap.CaptionRenderer(W, H, spec.caption_style, all_text)

    for si, sc in enumerate(scenes):
        lo, hi = 5 + 85 * si // n, 5 + 85 * (si + 1) // n
        progress(lo, f"Scene {si + 1}/{n}: generating voice")
        wav = workdir / f"s{si}.wav"
        engine = tts.synthesize(sc.text.strip(), spec.voice, spec.speed, wav)
        if si == 0 or engine != "edge":
            notes.append(f"Scene {si + 1} voice engine: {engine}")
        samples, sr = tts.wav_samples(wav)
        speech = len(samples) / sr
        total = max(speech, 0.8) + 0.4
        n_frames = math.ceil(total * fps)

        has_media = bool(sc.media_path and sc.media_path.exists())
        size, cx, bottom, media_box, cap_y = layout(spec.aspect_ratio, W, H, has_media)
        bg = make_background(sc.background, W, H, (bg_paths or {}).get(sc.background.get("asset_id") or ""))
        if has_media:
            _paint_media(bg, sc.media_path, media_box)
        base = bg
        ax, ay = cx - size // 2, bottom - size

        env = avatar_mod.amplitude_envelope(wav, fps, n_frames)
        s0, s1 = avatar_mod.speech_bounds(env, fps)
        words, joiner = cap.split_words(sc.text)
        timings = cap.word_timings(words, s0, max(s1, s0 + 0.5))
        chunks = cap.chunk_words(words)
        cap_cache: dict[tuple[int, int], Image.Image] = {}

        def caption_for(t: float):
            for ci, ch in enumerate(chunks):
                t_start, t_end = timings[ch[0]][0], timings[ch[-1]][1]
                if t_start - 0.05 <= t < (timings[chunks[ci + 1][0]][0] - 0.05 if ci + 1 < len(chunks) else total):
                    act = -1
                    if spec.caption_style == "bold":
                        for k in ch:
                            if timings[k][0] <= t < timings[k][1]:
                                act = k
                    key = (ci, act)
                    if key not in cap_cache:
                        cap_cache[key] = cap_r.render([words[k] for k in ch], joiner, ch.index(act) if act in ch else -1)
                    return cap_cache[key]
            return None

        progress(lo + (hi - lo) // 4, f"Scene {si + 1}/{n}: animating avatar")
        avatar_src = spec.avatar_path
        if avatar_src is None:
            raise RenderError("No avatar selected")
        frames_iter = provider.frames(avatar_src, spec.avatar_meta, wav, size, fps, n_frames, workdir)

        def raw_frames():
            nonlocal thumb_saved
            for i, av in enumerate(frames_iter):
                frame = base.copy()
                frame.paste(av, (ax, ay), av)
                if spec.captions_enabled:
                    c = caption_for(i / fps)
                    if c is not None:
                        frame.paste(c, (0, cap_y - c.height // 2), c)
                if not thumb_saved and i == min(n_frames - 1, fps // 2):
                    frame.save(out_thumb, "JPEG", quality=85)
                    thumb_saved = True
                yield frame.tobytes()

        clip = workdir / f"scene{si}.mp4"
        _run_ffmpeg([
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
            "-i", str(wav), "-af", "apad", "-t", f"{n_frames / fps:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2", "-movflags", "+faststart", str(clip),
        ], stdin_frames=raw_frames())
        clips.append(clip)
        progress(hi, f"Scene {si + 1}/{n}: done")

    progress(92, "Joining scenes")
    listing = workdir / "concat.txt"
    listing.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    joined = workdir / "joined.mp4"
    _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)])

    out_video.parent.mkdir(parents=True, exist_ok=True)
    if spec.music_path and spec.music_path.exists():
        progress(96, "Mixing background music")
        _run_ffmpeg([
            "-i", str(joined), "-stream_loop", "-1", "-i", str(spec.music_path),
            "-filter_complex", f"[1:a]volume={spec.music_volume:.2f}[m];[0:a][m]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]",
            "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest",
            "-movflags", "+faststart", str(out_video),
        ])
    else:
        joined.replace(out_video)
    progress(100, "Done")
    return RenderResult(out_video, out_thumb, probe_duration(out_video), notes)
