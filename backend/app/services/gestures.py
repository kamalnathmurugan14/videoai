"""Hand gestures for the puppet avatar.

A photo has no hands, so they are drawn: skin-coloured illustrated hands with sleeves sampled from the avatar's clothing.
Gestures are planned from the speech envelope: each spoken phrase raises a hand (or both) into an explaining / pointing /
counting pose, the hand "beats" with emphasis, then lowers out of frame during pauses.
"""
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Gesture:
    start: float
    end: float
    kind: str  # open | both | point | count2 | count3
    side: int  # +1 = right of screen, -1 = left


def _ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def plan(env: np.ndarray, fps: int, seed: int) -> list[Gesture]:
    """Find speech phrases in the envelope and assign a gesture to each."""
    rng = np.random.default_rng(seed + 101)
    active = np.convolve(env, np.ones(max(1, fps // 4)) / max(1, fps // 4), mode="same") > 0.07
    phrases, start, last = [], None, None
    for i, a in enumerate(active):
        if a:
            if start is None:
                start = i
            last = i
        elif start is not None and i - last > 0.32 * fps:
            phrases.append((start / fps, (last + 1) / fps))
            start = None
    if start is not None:
        phrases.append((start / fps, (last + 1) / fps))
    kinds = ["open", "both", "point", "open", "count2", "both", "count3", "open"]
    out, k, side = [], int(rng.integers(0, len(kinds))), int(rng.choice([-1, 1]))
    for a, b in phrases:
        if b - a < 0.9:
            continue
        n_pieces = max(1, round((b - a) / 2.3))  # a new gesture roughly every 2 s of continuous speech
        edges = np.linspace(a, b, n_pieces + 1)
        pieces = list(zip(edges[:-1], edges[1:]))
        for pa, pb in pieces:
            out.append(Gesture(pa + 0.12, pb - 0.05, kinds[k % len(kinds)], side))
            k += 1
            side = -side
    return out


# ----------------------------------------------------------------------------- drawing
def _composite(dst: Image.Image, src: Image.Image, x: int, y: int) -> None:
    """alpha_composite `src` onto `dst` at (x, y), clipping whatever falls outside."""
    cx0, cy0 = max(0, -x), max(0, -y)
    cx1, cy1 = min(src.width, dst.width - x), min(src.height, dst.height - y)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    dst.alpha_composite(src.crop((cx0, cy0, cx1, cy1)), (x + cx0, y + cy0))


def _capsule(d: ImageDraw.ImageDraw, p0, p1, w: float, fill) -> None:
    d.line([p0, p1], fill=fill, width=max(1, int(w)))
    r = w / 2
    for x, y in (p0, p1):
        d.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def _shade(c, f):
    return tuple(max(0, min(255, int(v * f))) for v in c[:3])


def hand_patch(skin, pose: str, pw: float, ss: int = 3) -> Image.Image:
    """Hand with fingers pointing up, palm centred. `pw` = palm width in output pixels. Returns RGBA."""
    u = pw * ss
    P = int(u * 2.6)
    im = Image.new("RGBA", (P, P), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, cy = P / 2, P / 2 + u * 0.35
    outline, fill, crease = _shade(skin, 0.62), _shade(skin, 1.0), _shade(skin, 0.8)
    ow = max(2, int(u * 0.05))
    extended = {"open": [1, 1, 1, 1], "point": [1, 0, 0, 0], "count2": [1, 1, 0, 0], "count3": [1, 1, 1, 0]}[pose]
    # (x offset, length, fan angle) per finger: index, middle, ring, little (thumb handled separately)
    spec = [(-0.30, 0.95, -8), (-0.10, 1.05, -2), (0.10, 0.98, 4), (0.30, 0.78, 10)]
    if pose == "open":
        spec = [(a, b, c * 1.8) for a, b, c in spec]
    fw = u * 0.215

    def finger(i, ext):
        x0, length, ang = spec[i]
        base = (cx + x0 * u, cy - 0.38 * u)
        L = length * u * (1.0 if ext else 0.26)
        a = math.radians(ang if ext else 0)
        tip = (base[0] + math.sin(a) * L, base[1] - math.cos(a) * L)
        for col, w in ((outline, fw + ow * 2), (fill, fw)):
            _capsule(d, base, tip, w, col)
        if not ext:  # curled finger: knuckle bump over the palm
            d.ellipse([base[0] - fw / 2, base[1] - fw * 0.15, base[0] + fw / 2, base[1] + fw * 0.85], fill=fill, outline=outline, width=ow)

    # fingers behind the palm first (so the palm overlaps their roots)
    for i in range(4):
        finger(i, extended[i])
    # palm
    d.rounded_rectangle([cx - 0.52 * u, cy - 0.45 * u, cx + 0.52 * u, cy + 0.60 * u], radius=int(0.28 * u), fill=outline)
    d.rounded_rectangle([cx - 0.52 * u + ow, cy - 0.45 * u + ow, cx + 0.52 * u - ow, cy + 0.60 * u - ow], radius=int(0.26 * u), fill=fill)
    # palm crease
    d.arc([cx - 0.3 * u, cy - 0.05 * u, cx + 0.3 * u, cy + 0.35 * u], 200, 340, fill=crease, width=max(1, ow // 2))
    # thumb
    tb = (cx - 0.46 * u, cy + 0.22 * u)
    tt = (cx - 0.95 * u, cy - 0.12 * u) if pose in ("open", "count3", "count2") else (cx - 0.55 * u, cy - 0.25 * u)
    for col, w in ((outline, fw * 1.15 + ow * 2), (fill, fw * 1.15)):
        _capsule(d, tb, tt, w, col)
    return im.resize((P // ss, P // ss), Image.LANCZOS)


class HandGestures:
    def __init__(self, size: int, skin, sleeve, env: np.ndarray, fps: int, seed: int):
        self.s = size
        self.skin = tuple(skin)
        self.sleeve = tuple(sleeve)
        self.fps = fps
        self.plan = plan(env, fps, seed)
        self.pw = 0.15 * size
        self._cache: dict[str, Image.Image] = {}
        self.env = env

    def _patch(self, pose: str) -> Image.Image:
        if pose not in self._cache:
            self._cache[pose] = hand_patch(self.skin, pose, self.pw)
        return self._cache[pose]

    # target pose for one hand: (x, y, angle_deg, pose) in normalised canvas coords at full raise
    def _targets(self, g: Gesture):
        s = g.side
        if g.kind == "both":
            return [(0.5 - 0.34, 0.76, 14, "open"), (0.5 + 0.34, 0.76, -14, "open")]
        if g.kind == "point":
            return [(0.5 + s * 0.34, 0.66, -s * 8, "point")]
        if g.kind in ("count2", "count3"):
            return [(0.5 + s * 0.33, 0.70, -s * 6, g.kind)]
        return [(0.5 + s * 0.33, 0.74, -s * 16, "open")]

    def draw(self, frame: Image.Image, t: float, body) -> None:
        """Draw active hands onto `frame` (RGBA, size x size). `body(x, y)` -> pixel offset of the torso at that point."""
        hands = []
        for g in self.plan:
            lo, hi = g.start - 0.45, g.end + 0.45
            if not (lo <= t <= hi):
                continue
            u = _ease(min((t - (g.start - 0.45)) / 0.45, (g.end + 0.45 - t) / 0.45, 1.0))  # raise, hold, lower
            if u <= 0.01:
                continue
            for i, (tx, ty, ang, pose) in enumerate(self._targets(g)):
                ph = g.start * 3.1 + i * 1.7
                beat = math.sin(2 * math.pi * 1.05 * t + ph)
                emph = float(self.env[min(len(self.env) - 1, int(t * self.fps))])
                x = tx + 0.008 * math.sin(2 * math.pi * 0.55 * t + ph)
                y = ty - 0.02 * emph * (0.5 + 0.5 * beat)
                rest_y = 1.18
                hy = rest_y + (y - rest_y) * u
                hands.append((x, hy, ang + 5 * beat * u, pose, u, i))
        for x, hy, ang, pose, u, _ in hands:
            self._draw_hand(frame, x, hy, ang, pose, body)

    def _draw_hand(self, frame, x, y, ang, pose, body) -> None:
        s = self.s
        bx, by = body(x, y)
        px, py = x * s + bx, y * s + by
        pw = self.pw
        # forearm: from just below the wrist, off the bottom of the canvas, angled slightly outward
        wrist = (px + math.sin(math.radians(-ang)) * pw * 0.75, py + math.cos(math.radians(ang)) * pw * 0.85)
        away = 1 if x > 0.5 else -1
        elbow = (wrist[0] + away * s * 0.07, s * 1.12)
        ss = 2
        x0 = int(min(px, wrist[0], elbow[0]) - pw * 1.6)
        x1 = int(max(px, wrist[0], elbow[0]) + pw * 1.6)
        y0 = int(py - pw * 2.0)
        y1 = int(s)
        if y1 <= y0 or x1 <= x0:
            return
        W, H = (x1 - x0) * ss, (y1 - y0) * ss
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        loc = lambda p: ((p[0] - x0) * ss, (p[1] - y0) * ss)
        sleeve_edge = _shade(self.sleeve, 0.7)
        # sleeve (forearm) ends a little below the wrist; skin wrist shows between sleeve and palm
        v = (elbow[0] - wrist[0], elbow[1] - wrist[1])
        n = math.hypot(*v) or 1
        cuff = (wrist[0] + v[0] / n * pw * 0.45, wrist[1] + v[1] / n * pw * 0.45)
        _capsule(d, loc(cuff), loc(elbow), pw * 1.05 * ss + 3 * ss, sleeve_edge)
        _capsule(d, loc(cuff), loc(elbow), pw * 1.05 * ss, self.sleeve)
        _capsule(d, loc((px, py + pw * 0.3)), loc(cuff), pw * 0.62 * ss + 3 * ss, _shade(self.skin, 0.62))
        _capsule(d, loc((px, py + pw * 0.3)), loc(cuff), pw * 0.62 * ss, self.skin)
        layer = layer.resize((x1 - x0, y1 - y0), Image.LANCZOS)
        _composite(frame, layer, x0, y0)
        patch = self._patch(pose).rotate(ang, resample=Image.BICUBIC, expand=False)
        _composite(frame, patch, int(px - patch.width / 2), int(py - patch.height / 2 + pw * 0.35))
