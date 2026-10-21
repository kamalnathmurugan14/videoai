"""Render pipeline: per scene TTS -> animated avatar frames -> captions -> FFmpeg; then concat + music."""
import itertools
import math
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from ..config import ASSETS_DIR, get_settings
from . import avatar as avatar_mod
from . import face as face_mod
from . import captions as cap
from . import remote_gpu, tts
from . import video_avatar as video_mod


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
    intro_enabled: bool = True
    intro_title: str = ""
    gestures_enabled: bool = True
    animation_engine: str = "builtin"  # builtin | sadtalker | gpu_echomimic | gpu_sadtalker | gpu_lipsync
    framing: str = "scene"  # scene: cut-out on a background | photo: original photo fills the frame (HeyGen style)
    photo_full_path: Path | None = None
    video_avatar: dict | None = None  # meta of a video avatar (real footage of the person, looped)


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
    if t == "scenery":
        f = ASSETS_DIR / "sceneries" / f"{(bg.get('scenery_id') or '').replace('/', '')}.jpg"
        if f.exists():
            return ImageOps.fit(Image.open(f).convert("RGB"), (w, h), Image.LANCZOS)
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


def _media_layer(path: Path, box, canvas: tuple[int, int]) -> tuple[Image.Image, tuple[int, int]]:
    """Rounded image card with drop shadow, pre-rendered once per scene. Returns (RGBA layer, top-left)."""
    x0, y0, x1, y1 = box
    card = ImageOps.contain(Image.open(path).convert("RGB"), (x1 - x0, y1 - y0), Image.LANCZOS)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    px, py = cx - card.width // 2, cy - card.height // 2
    pad = 40
    layer = Image.new("RGBA", (card.width + pad * 2, card.height + pad * 2), (0, 0, 0, 0))
    sh = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([pad + 4, pad + 10, pad + card.width + 4, pad + card.height + 10], 18, fill=(0, 0, 0, 110))
    layer.alpha_composite(sh.filter(ImageFilter.GaussianBlur(10)))
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, card.width - 1, card.height - 1], 18, fill=255)
    layer.paste(card, (pad, pad), mask)
    return layer, (px - pad, py - pad)


def _ease_out(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def _title_layer(title: str, w: int, h: int, aspect: str) -> tuple[Image.Image, int]:
    """Intro title text with soft shadow. Returns (RGBA layer of full width, centre y)."""
    size = max(18, int((0.075 * h) if aspect == "16:9" else (0.045 * h if aspect == "9:16" else 0.07 * h)))
    font = cap.get_font(title, size)
    d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    words, lines, cur = title.split(), [], ""
    for wd in words:
        trial = f"{cur} {wd}".strip()
        if cur and d0.textlength(trial, font=font) > w * 0.84:
            lines.append(cur)
            cur = wd
        else:
            cur = trial
    lines.append(cur)
    lh = int(size * 1.25)
    layer = Image.new("RGBA", (w, lh * len(lines) + 40), (0, 0, 0, 0))
    sh = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    for li, ln in enumerate(lines):
        x = (w - d0.textlength(ln, font=font)) / 2
        ImageDraw.Draw(sh).text((x + 3, 23 + li * lh + 4), ln, font=font, fill=(0, 0, 0, 170))
    layer.alpha_composite(sh.filter(ImageFilter.GaussianBlur(4)))
    d = ImageDraw.Draw(layer)
    for li, ln in enumerate(lines):
        d.text(((w - d0.textlength(ln, font=font)) / 2, 20 + li * lh), ln, font=font, fill=(255, 255, 255, 255), stroke_width=max(1, size // 20), stroke_fill=(20, 20, 40, 255))
    cy = int(h * (0.13 if aspect == "16:9" else 0.22 if aspect == "9:16" else 0.12))
    return layer, cy


def _with_lead_silence(wav: Path, lead: float) -> Path:
    """Prepend `lead` seconds of silence (the intro) to a mono 16-bit WAV, in place."""
    data, sr = tts.wav_samples(wav)
    pcm = np.concatenate([np.zeros(int(lead * sr), np.float32), data])
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    return wav


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


def _decode(video: Path, W: int, H: int, fps: int, n_frames: int):
    """Frames of an MP4 as W x H RGB images, exactly n_frames (the last frame repeats if the video is short)."""
    proc = subprocess.Popen([get_settings().ffmpeg_bin, "-loglevel", "error", "-i", str(video), "-vf", f"fps={fps},scale={W}:{H}",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    last = None
    try:
        for _ in range(n_frames):
            buf = proc.stdout.read(W * H * 3)
            if len(buf) == W * H * 3:
                last = Image.frombytes("RGB", (W, H), buf)
            if last is None:
                raise RenderError("the lip-sync result could not be decoded")
            yield last.copy()  # callers draw captions onto the frame
    finally:
        proc.stdout.close()
        proc.wait()


def render_video(spec: RenderSpec, workdir: Path, out_video: Path, out_thumb: Path, progress: Progress,
                 bg_paths: dict[str, Path] | None = None) -> RenderResult:
    s = get_settings()
    fps = s.render_fps
    W, H = dims(spec.aspect_ratio)
    workdir.mkdir(parents=True, exist_ok=True)
    provider = avatar_mod.get_provider(spec.animation_engine)
    notes: list[str] = []
    scenes = [sc for sc in spec.scenes if sc.text.strip()]
    if not scenes:
        raise RenderError("Add some script text to at least one scene")
    n = len(scenes)
    clips: list[Path] = []
    thumb_saved = False
    prev_last: Image.Image | None = None
    all_text = " ".join(sc.text for sc in scenes)
    cap_r = cap.CaptionRenderer(W, H, spec.caption_style, all_text)
    title = (spec.intro_title or "").strip()
    PAN = 1.12  # background is rendered 12% larger and drifts slowly for a sense of camera movement
    lipsync = spec.animation_engine == "gpu_lipsync"
    vavatar = None
    if spec.video_avatar and spec.animation_engine in ("builtin", "gpu_lipsync"):
        try:
            vavatar = video_mod.load(spec.video_avatar)
        except Exception as e:
            notes.append(f"⚠ Video avatar unavailable ({str(e)[:200]}); animated its photo instead")
    elif spec.video_avatar:
        notes.append(f"Video avatar: the {spec.animation_engine} engine works from one photo, so a still frame of your clip was used")
    if lipsync:
        provider.draw_mouth = False  # the GPU model draws the mouth; the built-in animation only supplies the motion
    photo_mode = bool(vavatar is None and spec.framing == "photo" and spec.photo_full_path and spec.photo_full_path.exists()
                      and (spec.avatar_meta or {}).get("full"))
    avatar_path, avatar_meta = spec.avatar_path, spec.avatar_meta
    if spec.framing == "photo" and not photo_mode and vavatar is None:
        notes.append("⚠ Original-photo framing needs an uploaded photo (re-upload older ones); used the cut-out instead")
    if photo_mode:  # HeyGen style: the original photo is the whole frame and is animated in place
        pc, avatar_meta = face_mod.photo_canvas(Image.open(spec.photo_full_path), spec.avatar_meta, W, H)
        avatar_path = workdir / "photo_canvas.png"
        pc.save(avatar_path)
        PF = avatar_meta["photo_frame"]
    XFADE = max(2, fps // 3)

    for si, sc in enumerate(scenes):
        lo, hi = 5 + 85 * si // n, 5 + 85 * (si + 1) // n
        progress(lo, f"Scene {si + 1}/{n}: generating voice")
        wav = workdir / f"s{si}.wav"
        engine, errors = tts.synthesize(sc.text.strip(), spec.voice, spec.speed, wav)
        note = f"Scene {si + 1} voice engine: {engine}"
        if errors:
            note = f"⚠ Scene {si + 1}: preferred voice engine failed ({errors[0]}); used '{engine}' instead"
        elif engine == "gtts" and tts.voice_gender(spec.voice) == "male":
            note = f"⚠ Scene {si + 1}: Google TTS has a single female voice, so the male voice you picked cannot be used"
        if note not in notes:
            notes.append(note)
        samples, sr = tts.wav_samples(wav)
        speech = len(samples) / sr
        lead = 2.3 if (si == 0 and spec.intro_enabled) else 0.0
        if lead:
            _with_lead_silence(wav, lead)
        total = lead + max(speech, 0.8) + 0.45
        n_frames = math.ceil(total * fps)

        has_media = bool(sc.media_path and sc.media_path.exists()) and not photo_mode and vavatar is None
        size, cx, bottom, media_box, cap_y = layout(spec.aspect_ratio, W, H, has_media)
        if photo_mode:
            size = max(W, H)
        bgW, bgH = int(W * PAN), int(H * PAN)
        bg_big = make_background(sc.background, bgW, bgH, (bg_paths or {}).get(sc.background.get("asset_id") or ""))
        pan_from, pan_to = ((0, 0), (bgW - W, bgH - H)) if si % 2 == 0 else ((bgW - W, 0), (0, bgH - H))
        media_layer = media_pos = None
        if has_media:
            media_layer, media_pos = _media_layer(sc.media_path, media_box, (W, H))
        title_layer = title_y = None
        if lead and title:
            title_layer, title_y = _title_layer(title, W, H, spec.aspect_ratio)
        ax, ay = cx - size // 2, bottom - size

        env = avatar_mod.amplitude_envelope(wav, fps, n_frames)
        s0, s1 = avatar_mod.speech_bounds(env, fps)
        words, joiner = cap.split_words(sc.text)
        timings = cap.word_timings(words, max(s0, lead), max(s1, max(s0, lead) + 0.5))
        chunks = cap.chunk_words(words)
        cap_cache: dict[tuple[int, int], Image.Image] = {}

        def caption_for(t: float):
            for ci, ch in enumerate(chunks):
                t_start = timings[ch[0]][0]
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

        stage = {"sadtalker": "realistic lip-sync with SadTalker (can take minutes)",
                 "gpu_echomimic": "generating on the Colab GPU with EchoMimicV2 (can take a long time)",
                 "gpu_sadtalker": "generating on the Colab GPU with SadTalker"}.get(provider.name, "animating avatar")
        if vavatar is not None:
            stage = "animating your video"
        progress(lo + (hi - lo) // 4, f"Scene {si + 1}/{n}: {stage}")
        if avatar_path is None:
            raise RenderError("No avatar selected")
        gestures_on = spec.gestures_enabled and not photo_mode and vavatar is None  # no drawn hands on real photos / video
        cursor0 = vavatar.cursor if vavatar is not None else 0.0

        def avatar_frames(mouth: bool):
            if vavatar is not None:
                vavatar.cursor = cursor0
                return vavatar.frames(wav, W, H, fps, n_frames, draw_mouth=mouth)
            prov = provider
            if mouth and not getattr(provider, "draw_mouth", True):
                prov = avatar_mod.PuppetAvatarProvider()
            return prov.frames(avatar_path, avatar_meta, wav, size, fps, n_frames, workdir, seed=si, gestures=gestures_on)

        frames_iter = avatar_frames(not lipsync)
        if vavatar is None and provider.name != avatar_mod.PuppetAvatarProvider.name:
            try:  # external engines do their heavy work on the first frame: fail over to the built-in animation
                first = next(frames_iter)
                frames_iter = itertools.chain([first], frames_iter)
                if getattr(provider, "last_message", ""):
                    notes.append(f"Scene {si + 1}: {provider.last_message}")
            except Exception as e:
                notes.append(f"⚠ Scene {si + 1}: {provider.name} failed ({str(e)[:300]}); used the built-in animation instead")
                frames_iter = avatar_mod.PuppetAvatarProvider().frames(avatar_path, avatar_meta, wav, size, fps, n_frames,
                                                                        workdir, seed=si, gestures=gestures_on)
        thumb_at = min(n_frames - 1, int((lead + 0.7) * fps)) if si == 0 else None

        def base_frames(frames_iter, entrance: bool):
            """Background + avatar (no titles / captions) as W x H RGB images."""
            for i, av in enumerate(frames_iter):
                t = i / fps
                if vavatar is not None:  # the (mouth-animated) footage is the whole frame
                    frame, av = av, None
                elif photo_mode:  # the animated photo is the whole frame
                    frame = av.crop((PF["x"], PF["y"], PF["x"] + W, PF["y"] + H)).convert("RGB")
                    av = None
                else:
                    k = i / max(1, n_frames - 1)
                    ox = round(pan_from[0] + (pan_to[0] - pan_from[0]) * k)
                    oy = round(pan_from[1] + (pan_to[1] - pan_from[1]) * k)
                    frame = bg_big.crop((ox, oy, ox + W, oy + H))
                if media_layer is not None:
                    frame.paste(media_layer, media_pos, media_layer)
                if av is not None:
                    # avatar entrance during the intro: rise + fade in
                    dy = 0
                    if si == 0 and lead and entrance:
                        e = _ease_out(t / 0.9)
                        dy = int((1 - e) * 0.12 * size)
                        if e < 1:
                            av = av.copy()
                            av.putalpha(av.getchannel("A").point(lambda v, e=e: int(v * e)))
                    frame.paste(av, (ax, ay + dy), av)
                yield frame

        bases = base_frames(frames_iter, entrance=not lipsync)
        if lipsync:  # send the moving (mouth-less) video to the GPU worker; LatentSync re-generates the lower face
            progress(lo + (hi - lo) // 4, f"Scene {si + 1}/{n}: lip-syncing on the Colab GPU (LatentSync)")
            base_mp4 = workdir / f"lipsync_in{si}.mp4"
            try:
                _run_ffmpeg(["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", str(base_mp4)],
                            stdin_frames=(f.tobytes() for f in bases))
                synced, msg = remote_gpu.render("lipsync", base_mp4, wav, workdir / f"lipsync_out{si}.mp4", fps)
                bases = _decode(synced, W, H, fps, n_frames)
                if msg:
                    notes.append(f"Scene {si + 1}: {msg}")
            except Exception as e:
                notes.append(f"⚠ Scene {si + 1}: GPU lip-sync failed ({str(e)[:300]}); used the built-in mouth animation instead")
                bases = base_frames(avatar_frames(True), entrance=True)

        def raw_frames():
            nonlocal thumb_saved, prev_last
            frame = None
            for i, frame in enumerate(bases):
                t = i / fps
                if title_layer is not None and t < lead:
                    ta = min(1.0, max(0.0, (t - 0.15) / 0.6)) * min(1.0, max(0.0, (lead - 0.1 - t) / 0.4))
                    if ta > 0:
                        tl = title_layer.copy()
                        tl.putalpha(tl.getchannel("A").point(lambda v, ta=ta: int(v * ta)))
                        frame.paste(tl, (0, title_y - tl.height // 2 + int((1 - _ease_out(t / 0.8)) * 18)), tl)
                if spec.captions_enabled:
                    c = caption_for(t)
                    if c is not None:
                        frame.paste(c, (0, cap_y - c.height // 2), c)
                if prev_last is not None and i < XFADE:
                    frame = Image.blend(prev_last, frame, (i + 1) / (XFADE + 1))
                if thumb_at is not None and not thumb_saved and i == thumb_at:
                    frame.save(out_thumb, "JPEG", quality=85)
                    thumb_saved = True
                yield frame.tobytes()
            prev_last = frame

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
