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


def draw_open_mouth(img: Image.Image, cx: float, cy: float, width: float, height: float, a: float) -> None:
    """Inner mouth for photos. The opening hangs below the lip seam (cx, cy): lens-shaped, with a shaded cavity,
    upper teeth and (when wide open) a tongue. Drawn supersampled and feathered so it blends with the photo's lips."""
    if height < 2.5:
        return
    ss = 3
    half = width * 0.5
    pad = int(width * 0.25) + 4
    pw, ph = int(width + pad * 2), int(height * 1.5 + pad * 2)
    n = 48
    us = np.linspace(-1, 1, n)
    env = np.clip(1 - us ** 2, 0, 1)
    top = (-height * 0.04 * env) * ss                      # almost flat upper edge, hugging the upper lip
    bot = (height * (env ** 0.5)) * ss                      # lower edge follows the dropped lower lip
    ox, oy = pw * ss / 2, (pad + height * 0.15) * ss        # lip seam sits near the top of the patch
    xs = ox + us * half * ss
    poly_top = [(x, oy + t) for x, t in zip(xs, top)]
    poly_bot = [(x, oy + b) for x, b in zip(xs[::-1], bot[::-1])]
    shape = poly_top + poly_bot
    layer = Image.new("RGBA", (pw * ss, ph * ss), (0, 0, 0, 0))
    cav = Image.new("L", layer.size, 0)
    ImageDraw.Draw(cav).polygon(shape, fill=255)
    # cavity: dark at the top (shadow under the upper lip) fading to warm red-brown at the bottom
    grad = np.zeros((ph * ss, pw * ss, 4), np.uint8)
    rows = np.arange(ph * ss, dtype=np.float32)
    t = np.clip((rows - (oy - height * 0.1 * ss)) / max(1.0, height * 1.1 * ss), 0, 1)[:, None]
    grad[..., 0] = (38 + 70 * t).astype(np.uint8)
    grad[..., 1] = (8 + 18 * t).astype(np.uint8)
    grad[..., 2] = (14 + 22 * t).astype(np.uint8)
    grad[..., 3] = 255
    layer = Image.fromarray(grad, "RGBA")
    d = ImageDraw.Draw(layer)
    if height > width * 0.05:  # upper teeth, tinted and shaded rather than a bright bar
        th = height * 0.30
        tx0, tx1 = ox - half * 0.62 * ss, ox + half * 0.62 * ss
        for k in range(int(th * ss)):
            shade = 1 - 0.28 * k / max(1, th * ss)
            d.line([(tx0 + k * 0.06, oy - height * 0.1 * ss + k), (tx1 - k * 0.06, oy - height * 0.1 * ss + k)],
                   fill=(int(214 * shade), int(206 * shade), int(194 * shade), 235))
    if a > 0.55 and height > width * 0.1:  # tongue
        d.ellipse([ox - half * 0.46 * ss, oy + height * 0.52 * ss, ox + half * 0.46 * ss, oy + height * 1.15 * ss], fill=(160, 62, 74, 255))
    mask = cav.filter(ImageFilter.GaussianBlur(0.9 * ss))
    layer.putalpha(mask)
    outline = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ImageDraw.Draw(outline).line(shape + [shape[0]], fill=(52, 16, 20, 150), width=max(1, int(1.4 * ss)))
    layer = Image.alpha_composite(layer, outline)
    layer = layer.resize((pw, ph), Image.LANCZOS)
    img.alpha_composite(layer, (int(cx - pw / 2), int(cy - oy / ss)))


# ---------------------------------------------------------------- provider interface
class AvatarProvider(ABC):
    name = "base"

    @abstractmethod
    def frames(self, image_path: Path, meta: dict, wav: Path, size: int, fps: int, n_frames: int,
               workdir: Path, seed: int = 0, gestures: bool = True) -> Iterator[Image.Image]:
        """Yield `n_frames` RGBA images of shape (size, size)."""


def _sstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / max(b - a, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


class PuppetAvatarProvider(AvatarProvider):
    name = "amplitude"
    G = 96  # resolution of the coarse displacement grid

    def _field(self, x, y, m, p):
        """Displacement (normalised units) at normalised points (x, y); `p` holds this frame's motion parameters."""
        mx, my, mw = m["mouth"]
        chin, fw = m["chin"], m["face_w"]
        # body: breathing + sway that grows toward the head, bottom edge stays put
        lift = np.clip((1.0 - y) / 0.7, 0, 1)
        dx = p["sway"] * lift ** 1.4
        dy = -p["breath"] * (1.0 - y)
        # head: rotation about the neck + translation, fading out over the shoulders
        pvx, pvy = mx, chin + 0.04
        wh = 1.0 - _sstep(pvy - 0.02, pvy + 0.13, y)
        th = math.radians(p["roll"])
        rx, ry = x - pvx, y - pvy
        dx = dx + wh * ((math.cos(th) - 1) * rx - math.sin(th) * ry + p["yaw"])
        dy = dy + wh * (math.sin(th) * rx + (math.cos(th) - 1) * ry + p["nod"])
        # shoulders: shrug on emphasis; upper body leans in toward the camera
        sh = _sstep(chin + 0.02, chin + 0.10, y) * (1 - _sstep(0.88, 1.0, y)) * _sstep(0.10, 0.20, np.abs(x - mx))
        dy = dy - p["shrug"] * 0.016 * sh - p["lean"] * 0.020 * np.clip((1.0 - y) / 0.6, 0, 1)
        # eyebrows lift
        ex = (m["eyes"][0][0] + m["eyes"][1][0]) / 2
        ey = (m["eyes"][0][1] + m["eyes"][1][1]) / 2
        dy = dy - p["brow"] * 0.012 * np.exp(-0.5 * (((x - ex) / (fw * 0.5)) ** 2 + ((y - (ey - 0.055)) / 0.025) ** 2))
        # blink (photos): pull the skin above each eye down over it, which keeps the real texture and lighting
        if p["blink"] > 0 and not m["has_mouth_art"]:
            for bx, by, br in m["eyes"]:
                g = np.exp(-0.5 * (((x - bx) / (br * 0.9)) ** 2 + ((y - (by - br * 0.1)) / (br * 0.7)) ** 2))
                dy = dy + p["blink"] * br * 0.85 * g
        # jaw drop (photos): region below the mouth line, narrowing toward the chin
        if p["jaw"] > 0 and not m["has_mouth_art"]:
            wy = _sstep(my - 0.012, my + 0.03, y) * (1.0 - _sstep(chin - 0.01, chin + 0.07, y))
            wx = np.exp(-0.5 * ((x - mx) / (fw * 0.34)) ** 2)
            dy = dy + wy * wx * p["jaw"]
            lip = np.exp(-0.5 * (((x - mx) / (fw * 0.22)) ** 2 + ((y - (my - 0.018)) / 0.012) ** 2))
            dy = dy - 0.10 * p["jaw"] * lip  # upper lip lifts slightly, revealing the teeth
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
        jaw_max = 0.115 * m["face_w"]
        photo = not m["has_mouth_art"]

        gain = size  # normalised -> pixels
        hands = None
        if gestures:
            region = np.asarray(base)[int(size * 0.68): int(size * 0.78), int(size * 0.25): int(size * 0.75)].reshape(-1, 4)
            region = region[region[:, 3] > 150]  # clothing just below the chin / above the faded bottom edge
            sleeve = tuple(int(v) for v in np.median(region[:, :3], axis=0)) if len(region) else (70, 80, 110)
            skin_rgb = tuple(int(c * 0.94) for c in _hex(m["skin"]))
            hands = gest_mod.HandGestures(size, skin_rgb, sleeve, env, fps, seed)

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
                "jaw": a * jaw_max,
                "blink": closed,
            }
            dxs, dys = self._field(gx, gy, m, p)
            D = np.dstack([dxs, dys]).astype(np.float32)
            D = cv2.resize(D, (size, size), interpolation=cv2.INTER_LINEAR)
            map_x = X - D[..., 0] * gain
            map_y = Y - D[..., 1] * gain
            out = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            alpha = out[..., 3:4].astype(np.float32)
            rgb = np.where(alpha > 0, out[..., :3].astype(np.float32) * 255.0 / np.maximum(alpha, 1.0), 0)
            frame = Image.fromarray(np.dstack([np.clip(rgb, 0, 255), alpha]).astype(np.uint8), "RGBA")

            def moved(px, py):
                dx, dy = self._field(np.float32(px), np.float32(py), m, p)
                return (px + float(dx)) * size, (py + float(dy)) * size

            mx, my, mw = m["mouth"]
            cx, cy = moved(mx, my)
            if photo:
                h_px = a * jaw_max * size * 0.68
                draw_open_mouth(frame, cx, cy, mw * size * (0.8 - 0.14 * a), h_px, a)  # mouth narrows slightly as it opens
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


def get_provider() -> AvatarProvider:
    return CommandAvatarProvider() if get_settings().avatar_provider == "command" else PuppetAvatarProvider()
