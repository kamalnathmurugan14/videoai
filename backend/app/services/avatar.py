"""Avatar animation. Providers produce one RGBA frame per video frame.

PuppetAvatarProvider (default, "amplitude"): a 2D puppet. Audio loudness drives the mouth / jaw, while smooth procedural
motion (breathing, body sway, head roll / nod / turn, eyebrows, blinks) comes from a displacement field applied with
cv2.remap. It works on the bundled illustrations (drawn mouth) and on photos (jaw stretch + inner mouth, using the
landmarks found at upload time).

CommandAvatarProvider: shells out to an external lip-sync model (SadTalker / Wav2Lip ...) configured via
AVATAR_PROVIDER=command and AVATAR_COMMAND. Add your own by subclassing AvatarProvider.
"""
import math
import shlex
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..config import get_settings
from . import gestures as gest_mod
from .tts import wav_samples

DEFAULT_META = {
    "mouth": [0.5, 0.58, 0.19], "eyes": [[0.39, 0.40, 0.04], [0.61, 0.40, 0.04]], "skin": "#e0ac8a",
    "has_mouth_art": False, "chin": 0.70, "face_w": 0.46,
}


# ---------------------------------------------------------------- audio analysis
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


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or len(x) == 0:
        return x
    k = np.ones(n, np.float32) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


# ---------------------------------------------------------------- drawing helpers (bundled illustrated avatars)
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
        d.pieslice([mx - w, top - w * 0.3, mx + w, bottom + w * 0.1], 0, 180, fill="#3b0d1a")
        d.rectangle([mx - w, top - w * 0.3, mx + w, top + w * 0.02], fill="#3b0d1a")
        d.rectangle([mx - w * 0.7, top - w * 0.02, mx + w * 0.7, top + open_h * 0.22], fill="#f8f4f0")  # teeth
        d.ellipse([mx - w * 0.5, bottom - open_h * 0.45, mx + w * 0.5, bottom + open_h * 0.1], fill="#e0707f")  # tongue
        d.arc([mx - w, top - w * 0.3, mx + w, bottom + w * 0.1], 0, 180, fill=lip, width=lw)
        d.line([mx - w, top - w * 0.15, mx + w, top - w * 0.15], fill=lip, width=lw)
    patch = patch.resize((pw, ph), Image.LANCZOS)
    img.alpha_composite(patch, (int(cx - pw / 2), int(cy - ph / 2)))


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def draw_eyelids(img: Image.Image, eyes, skins, size: int, closed: float, photo: bool) -> None:
    """Blink: lower a feathered skin-coloured lid over each eye. closed in [0,1]; skins = one RGB colour per eye."""
    if closed <= 0.02:
        return
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for (ex, ey, er), skin in zip(eyes, skins):
        x, y, r = ex * size, ey * size, er * size
        sr, sg, sb = skin
        lid = (int(sr * 0.93), int(sg * 0.91), int(sb * 0.89), 255)
        top = y - r * 0.5
        bottom = top + closed * r * 1.0
        d.ellipse([x - r * 0.9, top, x + r * 0.9, bottom], fill=lid)
        if closed > 0.6:  # lash line along the closed lid
            d.arc([x - r * 0.85, bottom - r * 0.45, x + r * 0.85, bottom + r * 0.12], 15, 165, fill=(35, 22, 18, 190), width=max(1, int(r * 0.1)))
    layer = layer.filter(ImageFilter.GaussianBlur(max(0.8, size * (0.0018 if photo else 0.0008))))
    img.alpha_composite(layer)


def lens_profile(us: np.ndarray, mouth_w: float, fw: float, size: int) -> np.ndarray:
    """Normalised (0..1) lower-edge profile of the opening across the mouth: rounded (half-ellipse) so it reads as an open
    mouth rather than a V. `us` in [-1, 1] spans the opening width."""
    return np.clip(1 - np.abs(us) ** 2.2, 0, 1) ** 0.62


def draw_open_mouth(img: Image.Image, cx: float, cy: float, width: float, height: float, a: float, fw: float, size: int,
                    top_offset=None, soft: float = 0.0, tint=None) -> None:
    """Inner mouth for photos. Fills exactly the gap the jaw warp opens below the lip seam (cx, cy): shaded cavity,
    upper teeth, tongue when wide, and a wet lower-lip highlight. Supersampled + feathered to blend with the photo.
    soft > 0 (real video footage): darker, low-contrast interior blurred by `soft` px to match the footage's sharpness;
    `tint` (the person's lip colour, RGB) colours the cavity and teeth so they match the footage's lighting / white balance."""
    if height < 2.0:
        return
    video = soft > 0
    ss = 3
    half = width * 0.5
    pad = int(width * 0.35) + 6
    pw, ph = int(width + pad * 2), int(height + pad * 2 + 8)
    n = 64
    us = np.linspace(-1, 1, n)
    prof = lens_profile(us, width, fw, size)
    oy = pad * ss
    ox = pw * ss / 2
    xs = ox + us * half * ss
    toff = np.zeros(n) if top_offset is None else np.array([top_offset(float(u)) for u in us])  # px, follows the lip curve
    top = [(x, oy - 0.6 * ss + t * ss) for x, t in zip(xs, toff)]
    bot = [(x, oy + (t + height * v) * ss) for x, v, t in zip(xs[::-1], prof[::-1], toff[::-1])]
    shape = top + bot
    W, H = pw * ss, ph * ss
    cav = Image.new("L", (W, H), 0)
    ImageDraw.Draw(cav).polygon(shape, fill=255)
    rows = np.arange(H, dtype=np.float32)
    t = np.clip((rows - oy) / max(1.0, height * ss), 0, 1)[:, None]
    arr = np.zeros((H, W, 4), np.uint8)
    lip = np.array(tint if tint is not None else (170, 100, 100), np.float32)
    if video:  # real mouths are nearly black inside, warming slightly toward the lower lip
        for ch in range(3):
            arr[..., ch] = np.clip(lip[ch] * (0.17 + 0.25 * t), 0, 255).astype(np.uint8)
    else:
        arr[..., 0] = (34 + 52 * t).astype(np.uint8)
        arr[..., 1] = (8 + 14 * t).astype(np.uint8)
        arr[..., 2] = (12 + 18 * t).astype(np.uint8)
    arr[..., 3] = 255
    layer = Image.fromarray(arr, "RGBA")
    feat = Image.new("RGBA", (W, H), (0, 0, 0, 0))  # teeth / tongue drawn separately so they can be softened
    d = ImageDraw.Draw(feat)
    teeth_k = 0.8 if video else 1.0
    if height > width * 0.06:  # upper teeth: a soft, shaded band that follows the curved top edge and fades into the cavity
        tprof = np.clip(1 - np.abs(us) ** 2.4, 0, 1) ** 0.5
        th = height * (0.45 if video else 0.34)
        for j in range(n - 1):
            x0, x1 = xs[j], xs[j + 1]
            hh = th * float(tprof[j]) * 0.82 * ss
            if hh < 1:
                continue
            for k in range(int(hh)):
                shade = 1 - 0.55 * (k / max(1.0, hh)) ** (3.0 if video else 1.3)
                tc = tuple(lip * 0.45 + np.array((236, 232, 224)) * 0.55 * min(1.0, float(lip.mean()) / 120)) if video else (176, 166, 152)
                d.line([(x0, oy + toff[j] * ss + k), (x1, oy + toff[j + 1] * ss + k)], fill=(int(tc[0] * shade), int(tc[1] * shade), int(tc[2] * shade), int((225 * shade + 10) * teeth_k)))
    if a > 0.75 and height > width * 0.14 and not video:  # tongue, dark and soft
        d.ellipse([ox - half * 0.38 * ss, oy + height * 0.52 * ss, ox + half * 0.38 * ss, oy + height * 0.92 * ss], fill=(118, 46, 56, 215))
    feat = feat.filter(ImageFilter.GaussianBlur(0.9 * ss))
    layer = Image.alpha_composite(layer, feat)
    layer.putalpha(cav.filter(ImageFilter.GaussianBlur(0.7 * ss)))
    edge = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ed = ImageDraw.Draw(edge)
    ed.line(bot[::-1], fill=(196, 120, 122, 70 if video else 150), width=max(1, int(1.6 * ss)))  # wet lower-lip edge
    ed.line(top, fill=(40, 12, 14, 90 if video else 170), width=max(1, int(1.3 * ss)))  # shadow under the upper lip
    layer = Image.alpha_composite(layer, edge.filter(ImageFilter.GaussianBlur(0.6 * ss)))
    layer = layer.resize((pw, ph), Image.LANCZOS)
    if video:
        layer = layer.filter(ImageFilter.GaussianBlur(soft))
    img.alpha_composite(layer, (int(cx - pw / 2), int(cy - oy / ss)))


# ---------------------------------------------------------------- provider interface
class AvatarProvider(ABC):
    name = "base"

    @abstractmethod
    def frames(self, image_path: Path, meta: dict, wav: Path, size: int, fps: int, n_frames: int,
               workdir: Path, seed: int = 0, gestures: bool = True) -> Iterator[Image.Image]:
        """Yield `n_frames` RGBA images of shape (size, size)."""


def _sstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / np.maximum(b - a, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def _make_hands(base: Image.Image, m: dict, size: int, env: np.ndarray, fps: int, seed: int):
    region = np.asarray(base)[int(size * 0.68): int(size * 0.78), int(size * 0.25): int(size * 0.75)].reshape(-1, 4)
    region = region[region[:, 3] > 150]  # clothing just below the chin / above the faded bottom edge
    sleeve = tuple(int(v) for v in np.median(region[:, :3], axis=0)) if len(region) else (70, 80, 110)
    skin_rgb = tuple(int(c * 0.94) for c in _hex(m["skin"]))
    return gest_mod.HandGestures(size, skin_rgb, sleeve, env, fps, seed)


class PuppetAvatarProvider(AvatarProvider):
    name = "amplitude"
    G = 96  # resolution of the coarse displacement grid
    draw_mouth = True  # False: head/body motion only (a GPU lip-sync model adds the mouth afterwards)

    def _field(self, x, y, m, p):
        """Displacement (normalised units) at normalised points (x, y); `p` holds this frame's motion parameters.
        Distances are tuned for a face 0.46 wide; `motion_scale` rescales them for smaller faces (full-frame photos)."""
        mx, my, mw = m["mouth"]
        chin, fw = m["chin"], m["face_w"]
        k = m.get("motion_scale", 1.0)
        # body: breathing + sway that grows toward the head, bottom edge stays put
        lift = np.clip((1.0 - y) / 0.7, 0, 1)
        dx = p["sway"] * k * lift ** 1.4
        dy = -p["breath"] * k * (1.0 - y)
        # head: rotation about the neck + translation, fading out over the shoulders
        pvx, pvy = mx, chin + 0.04 * k
        wh = 1.0 - _sstep(pvy - 0.02 * k, pvy + 0.13 * k, y)
        th = math.radians(p["roll"])
        rx, ry = x - pvx, y - pvy
        dx = dx + wh * ((math.cos(th) - 1) * rx - math.sin(th) * ry + p["yaw"] * k)
        dy = dy + wh * (math.sin(th) * rx + (math.cos(th) - 1) * ry + p["nod"] * k)
        # shoulders: shrug on emphasis; upper body leans in toward the camera
        sh = _sstep(chin + 0.02 * k, chin + 0.10 * k, y) * (1 - _sstep(0.88, 1.0, y)) * _sstep(0.10 * k, 0.20 * k, np.abs(x - mx))
        dy = dy - p["shrug"] * 0.016 * k * sh - p["lean"] * 0.020 * k * np.clip((1.0 - y) / 0.6, 0, 1)
        # eyebrows lift
        ex = (m["eyes"][0][0] + m["eyes"][1][0]) / 2
        ey = (m["eyes"][0][1] + m["eyes"][1][1]) / 2
        dy = dy - p["brow"] * 0.012 * k * np.exp(-0.5 * (((x - ex) / (fw * 0.5)) ** 2 + ((y - (ey - 0.055 * k)) / (0.025 * k)) ** 2))
        # blink (photos): pull the skin above each eye down over it, which keeps the real texture and lighting
        if p["blink"] > 0 and not m["has_mouth_art"]:
            for bx, by, br in m["eyes"]:
                g = np.exp(-0.5 * (((x - bx) / (br * 0.9)) ** 2 + ((y - (by - br * 0.1)) / (br * 0.7)) ** 2))
                dy = dy + p["blink"] * br * 0.85 * g
        # jaw drop (photos): everything below the lip seam shifts down as a block; narrow near the mouth (so the gap is
        # lens-shaped and tapers to the lip corners), plus a softer wide term so the whole chin follows
        if p["jaw"] > 0 and not m["has_mouth_art"]:
            seam_y = my
            if m.get("seam"):  # follow the real lip line (smiles curve up at the corners)
                sx, sy = zip(*m["seam"])
                seam_y = np.interp(x, sx, sy)
                outside = (x < min(sx)) | (x > max(sx))  # beyond the lip corners never start above the lip centre
                seam_y = np.where(outside, np.maximum(seam_y, my + 0.004 * k), seam_y)
            below = _sstep(seam_y, seam_y + 0.004 * k, y) * (1.0 - _sstep(chin - 0.01 * k, chin + 0.07 * k, y))
            narrow = np.exp(-0.5 * ((x - mx) / (fw * 0.20)) ** 2)
            wide = np.exp(-0.5 * ((x - mx) / (fw * 0.45)) ** 2)
            dy = dy + below * p["jaw"] * (0.75 * narrow + 0.25 * wide)
        return dx, dy

    def frames(self, image_path, meta, wav, size, fps, n_frames, workdir, seed=0, gestures=True):
        m = {**DEFAULT_META, **(meta or {})}
        base = Image.open(image_path).convert("RGBA")
        if base.width != base.height:  # legacy non-square upload: centre on a square canvas
            side = max(base.size)
            sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
            sq.paste(base, ((side - base.width) // 2, (side - base.height) // 2))
            base = sq
        base = base.resize((size, size), Image.LANCZOS)
        arr = np.asarray(base).astype(np.float32)
        arr[..., :3] *= arr[..., 3:4] / 255.0  # premultiply so remap doesn't fringe the cut-out edge
        src = arr.astype(np.uint8)
        X, Y = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))
        gy, gx = np.mgrid[0:self.G, 0:self.G].astype(np.float32) / (self.G - 1)

        env = amplitude_envelope(wav, fps, n_frames)
        emph = _smooth(env, max(2, fps // 5))  # syllable-scale energy
        talk = _smooth(env, fps)  # "is speaking" (≈1 s window)
        rng = np.random.default_rng(seed + 7)
        ph = rng.uniform(0, 2 * math.pi, 8)
        blink_period = 3.1 + rng.uniform(0, 1.4)
        jaw_max = 0.115 * m["face_w"] * (0.6 if m.get("open") else 1.0)  # smiling/open photos already show teeth: move less
        photo = not m["has_mouth_art"]

        gain = size  # normalised -> pixels
        hands = _make_hands(base, m, size, env, fps, seed) if gestures else None

        for i in range(n_frames):
            t = i / fps
            a = float(env[i])
            g = 0.55 + 0.9 * float(talk[i])  # livelier while speaking, calmer when idle
            e = float(emph[i])
            blink_phase = (t - 0.9) % blink_period
            closed = math.sin(math.pi * blink_phase / 0.17) if (t > 0.9 and blink_phase < 0.17) else 0.0
            p = {
                "roll": g * (2.4 * math.sin(2 * math.pi * 0.23 * t + ph[0]) + 1.2 * math.sin(2 * math.pi * 0.57 * t + ph[1])),
                "yaw": g * 0.011 * math.sin(2 * math.pi * 0.19 * t + ph[2]) + 0.004 * e * math.sin(2 * math.pi * 0.8 * t + ph[3]),
                "nod": g * 0.0075 * math.sin(2 * math.pi * 0.36 * t + ph[3]) + 0.011 * e,
                "shrug": float(np.clip(1.6 * (e - 0.45), 0, 1)) * (0.5 + 0.5 * math.sin(2 * math.pi * 0.7 * t + ph[5])),
                "lean": 0.5 * g + 0.8 * e * (0.5 + 0.5 * math.sin(2 * math.pi * 0.31 * t + ph[6])),
                "breath": 0.007 * math.sin(2 * math.pi * 0.27 * t + ph[4]),
                "sway": g * 0.012 * math.sin(2 * math.pi * 0.15 * t + ph[5]) + 0.003 * math.sin(2 * math.pi * 0.41 * t + ph[6]),
                "brow": float(np.clip(1.4 * (e - 0.35), 0, 1)) * (0.6 + 0.4 * math.sin(2 * math.pi * 0.9 * t + ph[7])),
                "jaw": a * jaw_max if self.draw_mouth else 0.0,
                "blink": closed,
            }
            dxs, dys = self._field(gx, gy, m, p)
            D = np.dstack([dxs, dys]).astype(np.float32)
            D = cv2.resize(D, (size, size), interpolation=cv2.INTER_LINEAR)
            map_x = X - D[..., 0] * gain
            map_y = Y - D[..., 1] * gain
            border = cv2.BORDER_REPLICATE if m.get("photo_frame") else cv2.BORDER_CONSTANT  # opaque photo: extend the edges
            out = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR, borderMode=border, borderValue=(0, 0, 0, 0))
            alpha = out[..., 3:4].astype(np.float32)
            rgb = np.where(alpha > 0, out[..., :3].astype(np.float32) * 255.0 / np.maximum(alpha, 1.0), 0)
            frame = Image.fromarray(np.dstack([np.clip(rgb, 0, 255), alpha]).astype(np.uint8), "RGBA")

            def moved(px, py):
                dx, dy = self._field(np.float32(px), np.float32(py), m, p)
                return (px + float(dx)) * size, (py + float(dy)) * size

            mx, my, mw = m["mouth"]
            cx, cy = moved(mx, my)
            if not self.draw_mouth:
                pass
            elif photo:
                h_px = a * jaw_max * size + 0.004 * m.get("motion_scale", 1.0) * size * (1 if a > 0.02 else 0)  # the gap the warp opens
                width_px = mw * size * (1.0 - 0.10 * a)
                off = None
                if m.get("seam"):
                    sx, sy = zip(*m["seam"])
                    base_y = float(np.interp(mx, sx, sy))
                    off = lambda u, w=width_px: (float(np.interp(mx + u * w / 2 / size, sx, sy)) - base_y) * size
                draw_open_mouth(frame, cx, cy, width_px, h_px, a, m["face_w"], size, off)
            else:
                draw_mouth(frame, cx, cy, mw * size, a)
            if closed > 0 and m["eyes"]:
                eyes = [(*moved(ex, ey), er * size) for ex, ey, er in m["eyes"]]
                if not photo:  # photos blink via the warp above; illustrations get a drawn lid
                    draw_eyelids(frame, [(x / size, y / size, r / size) for x, y, r in eyes], [_hex(m["skin"])] * 2, size, closed, photo)
            if hands is not None:
                hands.draw(frame, t, lambda hx, hy: tuple(float(v) * size for v in self._field(np.float32(hx), np.float32(hy), m, p)))
            yield frame


AmplitudeAvatarProvider = PuppetAvatarProvider  # backwards-compatible name


class CommandAvatarProvider(AvatarProvider):
    """Runs AVATAR_COMMAND ({image} {audio} {out} placeholders) to produce a talking-head video, then reads frames.
    Output is composited as an opaque square (most lip-sync models do not output alpha)."""

    name = "command"

    def frames(self, image_path, meta, wav, size, fps, n_frames, workdir, seed=0, gestures=True):
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


def _engine_input(image_path: Path, m: dict, workdir: Path, seed: int) -> tuple[Path, Image.Image]:
    """Image to hand to an external video model, plus the square canvas it came from. Original-photo framing: exactly
    the framed photo (its own background). Cut-out framing: the cut-out on neutral grey (easy to segment back out)."""
    canvas = Image.open(image_path).convert("RGBA")
    if canvas.width != canvas.height:
        side = max(canvas.size)
        sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        sq.paste(canvas, ((side - canvas.width) // 2, (side - canvas.height) // 2))
        canvas = sq
    pf = m.get("photo_frame")
    if pf:
        src_img = canvas.crop((pf["x"], pf["y"], pf["x"] + pf["W"], pf["y"] + pf["H"])).convert("RGB")
    else:
        src_img = Image.new("RGB", canvas.size, (128, 128, 128))
        src_img.paste(canvas, (0, 0), canvas)
        src_img = src_img.resize((512, 512), Image.LANCZOS)
    src = workdir / f"engine_src_{seed}.png"
    src_img.save(src)
    return src, canvas


def _frames_from_video(video: Path, canvas: Image.Image, m: dict, wav: Path, size: int, fps: int, n_frames: int,
                       seed: int, gestures: bool) -> Iterator[Image.Image]:
    """Decode an engine's output into `n_frames` RGBA frames of size x size, matching the built-in provider's contract."""
    from . import segment

    pf = m.get("photo_frame")
    base = canvas.resize((size, size), Image.LANCZOS)
    if pf:  # output covers the framed photo region of the (opaque) square canvas
        k = size / canvas.width
        fx, fy, fw_, fh_ = round(pf["x"] * k), round(pf["y"] * k), round(pf["W"] * k), round(pf["H"] * k)
        vw, vh = fw_ - fw_ % 2, fh_ - fh_ % 2
    else:
        vw = vh = size
        kk = max(3, int(size * 0.04)) | 1
        allowed = cv2.dilate(np.asarray(base.getchannel("A")), np.ones((kk, kk), np.uint8)).astype(np.float32) / 255
    env = amplitude_envelope(wav, fps, n_frames)
    hands = _make_hands(base, m, size, env, fps, seed) if (gestures and not pf) else None
    proc = subprocess.Popen(
        [get_settings().ffmpeg_bin, "-loglevel", "error", "-i", str(video), "-vf", f"fps={fps},scale={vw}:{vh}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    nbytes = vw * vh * 3
    last = None
    try:
        for i in range(n_frames):
            buf = proc.stdout.read(nbytes)
            if len(buf) == nbytes:
                rgb = np.frombuffer(buf, np.uint8).reshape(vh, vw, 3)
                if pf:
                    frame = base.copy()
                    frame.paste(Image.fromarray(rgb), (fx, fy))
                    last = frame
                else:
                    prob = segment.person_probability(rgb)
                    alpha = (np.clip((prob - 0.3) / 0.4, 0, 1) if prob is not None else allowed) * allowed
                    last = Image.fromarray(np.dstack([rgb, (alpha * 255).astype(np.uint8)]), "RGBA")
            if last is None:
                raise RuntimeError("the engine's output video could not be decoded")
            frame = last.copy() if hands is not None else last
            if hands is not None:
                hands.draw(frame, i / fps, lambda hx, hy: (0.0, 0.0))
            yield frame
    finally:
        proc.stdout.close()
        proc.wait()


class SadTalkerAvatarProvider(AvatarProvider):
    """Neural lip-sync via a local SadTalker install (see setup-sadtalker.ps1)."""

    name = "sadtalker"

    def frames(self, image_path, meta, wav, size, fps, n_frames, workdir, seed=0, gestures=True):
        from . import sadtalker

        m = {**DEFAULT_META, **(meta or {})}
        src, canvas = _engine_input(image_path, m, workdir, seed)
        video = sadtalker.run(src, wav, workdir / f"sadtalker_{seed}")
        yield from _frames_from_video(video, canvas, m, wav, size, fps, n_frames, seed, gestures)


class RemoteGPUProvider(AvatarProvider):
    """Runs a neural engine on a remote GPU worker (the free Google Colab notebook): EchoMimicV2 (half body with hand
    gestures, closest to HeyGen) or SadTalker (face)."""

    def __init__(self, engine: str):
        self.engine = engine
        self.name = f"gpu_{engine}"

    def frames(self, image_path, meta, wav, size, fps, n_frames, workdir, seed=0, gestures=True):
        from . import remote_gpu

        m = {**DEFAULT_META, **(meta or {})}
        src, canvas = _engine_input(image_path, m, workdir, seed)
        video, msg = remote_gpu.render(self.engine, src, wav, workdir / f"gpu_{seed}.mp4", fps)
        self.last_message = msg
        yield from _frames_from_video(video, canvas, m, wav, size, fps, n_frames, seed, gestures)


def get_provider(engine: str | None = None) -> AvatarProvider:
    if get_settings().avatar_provider == "command":
        return CommandAvatarProvider()
    if engine == "sadtalker":
        return SadTalkerAvatarProvider()
    if engine in ("gpu_echomimic", "gpu_sadtalker"):
        return RemoteGPUProvider(engine[4:])
    return PuppetAvatarProvider()
