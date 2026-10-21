"""Video avatars (HeyGen-style "real person" avatars): the user uploads a 10-20 s clip of themselves. The real footage
is looped (forward, then backward, so there is never a jump) for as long as the voice lasts, which keeps the person's
real head, body and blink movement. Only the mouth is re-animated to the new voice:

* built-in (CPU): per frame, the footage's own lips are closed with a warp, then the jaw is opened by the audio loudness
  and an inner mouth is drawn, following the tracked face (position, size and head tilt) in every frame;
* gpu_lipsync: the looped footage + voice go to the Colab GPU worker, where LatentSync regenerates the lower face.

At upload the clip is normalised to 25 fps JPEG frames and the face is tracked once per frame (Face Mesh); the
per-frame lip / eye / chin positions are stored next to the frames, so rendering needs no detection.
"""
import json
import math
import subprocess
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from fastapi import HTTPException
from PIL import Image

from ..config import get_settings
from . import face, landmarks, storage

SRC_FPS = 25
MAX_SECONDS = 20.0
MIN_SECONDS = 2.0
MAX_SIDE = 1280
FMT = "%05d.jpg"


# ----------------------------------------------------------------------------- upload / processing
def _probe(path: Path) -> tuple[float, bool, int]:
    ffprobe = get_settings().ffmpeg_bin.replace("ffmpeg", "ffprobe")
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type,width,height:format=duration",
                        "-of", "json", str(path)], capture_output=True, text=True)
    try:
        info = json.loads(r.stdout or "{}")
    except ValueError:
        info = {}
    vs = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    has_video = bool(vs)
    long_side = max(int(vs[0].get("width") or 0), int(vs[0].get("height") or 0)) if vs else 0
    try:
        dur = float(info.get("format", {}).get("duration", 0))
    except (TypeError, ValueError):
        dur = 0.0
    return dur, has_video, long_side


def _ffmpeg(args: list[str]) -> None:
    r = subprocess.run([get_settings().ffmpeg_bin, "-y", "-loglevel", "error", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise HTTPException(400, f"Could not read the video ({r.stderr.strip()[-200:] or 'ffmpeg failed'})")


def _track(frame_files: list[Path]) -> dict:
    """Face Mesh on every frame (tracked from the previous frame, re-detected when lost). Returns per-frame arrays."""
    n = len(frame_files)
    upper = np.zeros((n, 11, 2), np.float32)
    lower = np.zeros((n, 11, 2), np.float32)
    eyes = np.zeros((n, 2, 3), np.float32)
    chin = np.zeros((n, 2), np.float32)
    face_w = np.zeros(n, np.float32)
    yaw = np.zeros(n, np.float32)
    found = np.zeros(n, bool)
    prev = None
    for i, f in enumerate(frame_files):
        rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        lm = landmarks.track_landmarks(rgb, prev) if prev is not None else None
        if lm is None:
            near = tuple(prev.mean(0)) if prev is not None else None
            lm = landmarks.detect_landmarks(rgb, near)
        if lm is None:
            prev = None
            continue
        prev = lm["pts"]
        sm = landmarks.summarize(lm)
        upper[i], lower[i] = sm["upper_inner"], sm["lower_inner"]
        eyes[i] = np.array(sm["eyes"], np.float32)
        chin[i] = sm["chin"]
        face_w[i] = sm["face_w"]
        yaw[i] = sm["yaw"]
        found[i] = True
    return {"upper": upper, "lower": lower, "eyes": eyes, "chin": chin, "face_w": face_w, "yaw": yaw, "found": found}


def _fill_and_smooth(tr: dict, sigma: float = 1.0) -> dict:
    """Interpolate frames where the face was lost, then smooth over time (removes landmark jitter, keeps real motion)."""
    found = tr["found"]
    idx = np.arange(len(found))
    out = {"found": found}
    k = np.exp(-0.5 * (np.arange(-3, 4) / sigma) ** 2)
    k /= k.sum()
    for key in ("upper", "lower", "eyes", "chin", "face_w", "yaw"):
        a = tr[key].reshape(len(found), -1).astype(np.float64)
        for c in range(a.shape[1]):
            a[:, c] = np.interp(idx, idx[found], a[found, c])
            a[:, c] = np.convolve(np.pad(a[:, c], 3, mode="edge"), k, mode="valid")
        out[key] = a.reshape(tr[key].shape).astype(np.float32)
    return out


def mouth_gap(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Per-frame inner-lip gap at the centre divided by mouth width (0 = closed)."""
    w = np.linalg.norm(upper[..., -1, :] - upper[..., 0, :], axis=-1)
    return (lower[..., 5, 1] - upper[..., 5, 1]) / np.maximum(w, 1e-3)


def ingest(data: bytes, gender: str) -> tuple[str, int, int, dict]:
    """Uploaded clip -> frames + face track + a poster avatar (so still-image engines also work with it).
    Returns (poster filename, poster w, poster h, meta) like media.save_avatar."""
    s = get_settings()
    if len(data) > s.max_video_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Video too large (max {s.max_video_upload_mb} MB). Trim it to 10-15 seconds.")
    if landmarks._interpreters() is None:
        raise HTTPException(503, "Video avatars need the face tracking models (ai-edge-litert); see System check")
    name = storage.new_id()
    up = storage.subdir("uploads")
    tmp = storage.subdir("tmp") / f"{name}.upload"
    tmp.write_bytes(data)
    frames_dir = up / f"{name}_frames"
    try:
        dur, has_video, orig_long = _probe(tmp)
        if not has_video:
            raise HTTPException(400, "Not a video file (use MP4, MOV or WebM)")
        if 0 < dur < MIN_SECONDS:
            raise HTTPException(400, f"The clip is only {dur:.1f} s; record 10-15 seconds of yourself")
        frames_dir.mkdir()
        _ffmpeg(["-i", str(tmp), "-t", str(MAX_SECONDS), "-an", "-vf",
                 f"fps={SRC_FPS},scale={MAX_SIDE}:{MAX_SIDE}:force_original_aspect_ratio=decrease:force_divisible_by=2",
                 "-q:v", "3", str(frames_dir / FMT)])
        files = sorted(frames_dir.glob("*.jpg"))
        if len(files) < MIN_SECONDS * SRC_FPS:
            raise HTTPException(400, "Could not read enough frames from the video")
        raw = _track(files)
        found_ratio = float(raw["found"].mean())
        if found_ratio < 0.7:
            raise HTTPException(400, f"A face was found in only {found_ratio:.0%} of the clip. Record facing the camera, "
                                     "with your whole face visible and good light.")
        tr = _fill_and_smooth(raw)
        np.savez_compressed(up / f"{name}_track.npz", **tr)
        _ffmpeg(["-framerate", str(SRC_FPS), "-i", str(frames_dir / FMT), "-vf",
                 "scale=480:480:force_original_aspect_ratio=decrease:force_divisible_by=2", "-c:v", "libx264", "-crf", "28",
                 "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(up / f"{name}_preview.mp4")])
        gaps = mouth_gap(tr["upper"], tr["lower"])
        # poster: the frame with the most closed mouth and the most frontal face
        score = np.abs(gaps) + 0.5 * np.abs(tr["yaw"]) + (~raw["found"]) * 10
        best = int(np.argmin(score))
        poster = Image.open(files[best]).convert("RGB")
        canvas, meta, full = face.build_avatar_full(poster, True, gender)
        canvas.save(up / f"{name}.png", "PNG", optimize=True)
        full.save(up / f"{name}_full.jpg", "JPEG", quality=92)
        meta["full"]["file"] = f"{name}_full.jpg"
        h, w = cv2.imread(str(files[0])).shape[:2]
        warn = []
        if float(np.mean(gaps > 0.12)) > 0.4:
            warn.append("You are talking or smiling with an open mouth in much of the clip; the mouth is re-animated, but "
                        "a clip with relaxed, closed lips looks most natural.")
        if float(np.mean(np.abs(tr["yaw"]) > 0.3)) > 0.3:
            warn.append("The head is turned away from the camera in much of the clip; look into the camera for best results.")
        if found_ratio < 0.95:
            warn.append(f"The face was lost in {1 - found_ratio:.0%} of the frames (fast movement, hands or hair over the face).")
        meta["warning"] = " ".join(warn) or None
        meta["video"] = {"frames": f"{name}_frames", "track": f"{name}_track.npz", "preview": f"{name}_preview.mp4",
                         "n": len(files), "fps": SRC_FPS, "w": int(w), "h": int(h), "seconds": round(len(files) / SRC_FPS, 2), "orig_long": orig_long or max(w, h),
                         "poster_frame": best}
        return f"{name}.png", canvas.width, canvas.height, meta
    except BaseException:
        storage.rmtree(frames_dir)
        for suffix in ("_track.npz", "_preview.mp4", ".png", "_full.jpg"):
            storage.remove_rel(f"uploads/{name}{suffix}")
        raise
    finally:
        tmp.unlink(missing_ok=True)


def remove_files(meta: dict | None) -> None:
    v = (meta or {}).get("video")
    if not v:
        return
    up = storage.subdir("uploads")
    if v.get("frames") and "/" not in v["frames"] and ".." not in v["frames"]:
        storage.rmtree(up / v["frames"])
    for k in ("track", "preview"):
        storage.remove_rel(f"uploads/{v.get(k)}" if v.get(k) else None)


# ----------------------------------------------------------------------------- rendering
def _sstep(a, b, x):
    t = np.clip((x - a) / np.maximum(b - a, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


class VideoAvatar:
    """Loops the uploaded footage to any length and re-animates the mouth. `frames()` yields W x H RGB images."""

    def __init__(self, frames_dir: Path, track_path: Path, meta: dict):
        self.dir = frames_dir
        self.files = sorted(frames_dir.glob("*.jpg"))
        if len(self.files) < 2:
            raise RuntimeError("the video avatar's frames are missing; upload the clip again")
        with np.load(track_path) as z:
            self.tr = {k: z[k] for k in z.files}
        self.n = min(len(self.files), len(self.tr["face_w"]))
        h, w = cv2.imread(str(self.files[0])).shape[:2]
        self.w, self.h = w, h
        # footage sharpness: blur the drawn mouth like the footage around it (upscaled / soft clips need more)
        self.soft = 0.004 * max(1.0, max(w, h) / max(1, int(meta.get("video", {}).get("orig_long", max(w, h)))))
        self.cursor = 0.0  # seconds of footage already used, so consecutive scenes continue the motion

    # -- timing: forward, then backward, then forward ... (ping-pong) so the loop never jumps
    def index_at(self, t: float) -> int:
        span = (self.n - 1) / SRC_FPS
        tt = t % (2 * span)
        if tt > span:
            tt = 2 * span - tt
        return int(min(self.n - 1, round(tt * SRC_FPS)))

    # -- framing: cover the output frame with the face in the upper third (blurred fill if the shapes differ a lot)
    def _framer(self, W: int, H: int):
        eyes = self.tr["eyes"][: self.n]
        ex = float(np.median(eyes[:, :, 0]))
        ey = float(np.median(eyes[:, :, 1]))
        w, h = self.w, self.h
        if (W / H) / (w / h) > 1.3:  # e.g. portrait clip in a 16:9 video: fit height, blurred sides
            s = H / h
            nw = max(2, round(w * s))
            ramp = np.clip(np.minimum(np.arange(nw), nw - 1 - np.arange(nw)) / max(1, nw * 0.10), 0, 1) ** 1.5
            mask = np.tile(ramp, (H, 1))[..., None].astype(np.float32)
            x0 = (W - nw) // 2

            def fit(rgb):
                fg = cv2.resize(rgb, (nw, H), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
                small = cv2.resize(rgb, (max(2, W // 12), max(2, H // 12)), interpolation=cv2.INTER_AREA)
                bg = cv2.resize(cv2.GaussianBlur(cv2.resize(small, (W // 12 or 1, H // 12 or 1)), (0, 0), 2), (W, H))
                bg = (bg * 0.8).astype(np.float32)
                region = bg[:, x0:x0 + nw]
                bg[:, x0:x0 + nw] = region * (1 - mask) + fg.astype(np.float32) * mask
                return bg.astype(np.uint8)
            return fit
        s = max(W / w, H / h)
        nw, nh = max(W, round(w * s)), max(H, round(h * s))
        ox = int(np.clip(ex * s - W / 2, 0, nw - W))
        oy = int(np.clip(ey * s - 0.36 * H, 0, nh - H))

        def cover(rgb):
            big = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            return big[oy:oy + H, ox:ox + W]
        return cover

    # -- one frame of mouth animation in source pixels
    def animate(self, rgb: np.ndarray, i: int, a: float) -> np.ndarray:
        from .avatar import draw_open_mouth

        tr = self.tr
        up, lo = tr["upper"][i].astype(np.float32), tr["lower"][i].astype(np.float32)
        fw = float(tr["face_w"][i])
        F = fw * 0.46 / 0.39  # the puppet's face-width unit (its constants are tuned for 0.46 <-> mesh width 0.39)
        (ax, ay, _), (bx, by, _) = tr["eyes"][i]
        th = math.atan2(by - ay, bx - ax)
        c, s = math.cos(th), math.sin(th)
        chin = tr["chin"][i]
        H, W = rgb.shape[:2]
        mx, my = float((up[0, 0] + up[-1, 0]) / 2), float((up[5, 1] + lo[5, 1]) / 2)
        x0, x1 = int(max(0, mx - 0.85 * F)), int(min(W, mx + 0.85 * F))
        y0 = int(max(0, min(up[:, 1].min(), lo[:, 1].min()) - 0.35 * F))
        y1 = int(min(H, max(chin[1], lo[:, 1].max()) + 0.4 * F))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return rgb
        roi = rgb[y0:y1, x0:x1].copy()
        off = np.array([x0, y0], np.float32)
        up_l, lo_l = up - off, lo - off
        gap = float(mouth_gap(up, lo))
        if gap > 0.035:  # the footage's own mouth is open: close it first
            roi, _, seam = face.close_mouth(roi, up_l, lo_l)
        else:
            seam = face.seam_curve(up_l, lo_l)
        if a <= 0.02:
            out = rgb.copy()
            out[y0:y1, x0:x1] = roi
            return out
        # face-aligned coordinates (u along the eye line, v down the face), origin at the mouth centre
        mcx, mcy = mx - x0, my - y0
        rot = lambda px, py: (c * (px - mcx) + s * (py - mcy), -s * (px - mcx) + c * (py - mcy))
        su, sv = rot(seam[:, 0], seam[:, 1])
        order = np.argsort(su)
        su, sv = su[order], sv[order]
        _, chin_v = rot(chin[0] - x0, chin[1] - y0)
        jaw_max = 0.095 * F  # a little less than the photo puppet: real speech rarely opens wide
        jaw = a * jaw_max
        hh, ww = roi.shape[:2]
        X, Y = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(hh, dtype=np.float32))
        U, V = rot(X, Y)
        seam_v = np.interp(U, su, sv)
        beyond = np.maximum(su[0] - U, U - su[-1]).clip(0)  # distance past the lip corners
        seam_v = np.where(beyond > 0, np.maximum(seam_v, float(np.interp(0, su, sv)) + 0.0087 * F), seam_v)
        ramp = 0.0087 * F + 0.6 * beyond  # past the corners the skin stretches over a widening band (no crease line)
        below = _sstep(seam_v, seam_v + ramp, V) * (1 - _sstep(chin_v - 0.022 * F, chin_v + 0.15 * F, V))
        narrow = np.exp(-0.5 * (U / (0.20 * F)) ** 2)
        wide = np.exp(-0.5 * (U / (0.45 * F)) ** 2)
        edge = (1 - _sstep(0.55 * F, 0.82 * F, np.abs(U)))  # nothing moves at the ROI border
        d = (below * jaw * (0.75 * narrow + 0.25 * wide) * edge).astype(np.float32)
        map_x = X + d * s  # backward map: sample from "up the face"
        map_y = Y - d * c
        roi = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # inner mouth, drawn upright then rotated to the head's tilt
        width_px = (su[-1] - su[0]) * 1.06 * (1.0 - 0.10 * a)
        h_px = jaw + 0.0087 * F
        cu = (su[0] + su[-1]) / 2
        base_v = float(np.interp(cu, su, sv))
        top_offset = lambda u, w=width_px: float(np.interp(cu + u * w / 2, su, sv)) - base_v
        layer = Image.new("RGBA", (ww, hh), (0, 0, 0, 0))
        ccx, ccy = mcx + c * cu - s * base_v, mcy + s * cu + c * base_v  # back to ROI pixels
        lip_px = roi[np.clip(lo_l[2:9, 1].astype(int) + 3, 0, hh - 1), np.clip(lo_l[2:9, 0].astype(int), 0, ww - 1)]
        tint = tuple(float(v) for v in np.median(lip_px.reshape(-1, 3), axis=0))  # lower-lip colour in this light
        draw_open_mouth(layer, ccx, ccy, width_px, h_px, a, F, max(ww, hh), top_offset, soft=self.soft * F, tint=tint)
        if abs(th) > 0.01:
            layer = layer.rotate(-math.degrees(th), resample=Image.BICUBIC, center=(ccx, ccy))
        base = Image.fromarray(roi).convert("RGBA")
        base.alpha_composite(layer)
        out = rgb.copy()
        out[y0:y1, x0:x1] = np.asarray(base.convert("RGB"))
        return out

    def frames(self, wav: Path, W: int, H: int, fps: int, n_frames: int, draw_mouth: bool = True) -> Iterator[Image.Image]:
        from .avatar import amplitude_envelope

        env = amplitude_envelope(wav, fps, n_frames) if draw_mouth else np.zeros(n_frames, np.float32)
        fit = self._framer(W, H)
        start = self.cursor
        cache: tuple[int, np.ndarray] | None = None
        for k in range(n_frames):
            i = self.index_at(start + k / fps)
            if cache is None or cache[0] != i:
                cache = (i, cv2.cvtColor(cv2.imread(str(self.files[i])), cv2.COLOR_BGR2RGB))
            rgb = cache[1]
            if draw_mouth:
                rgb = self.animate(rgb, i, float(env[k]))
            yield Image.fromarray(fit(rgb))
        self.cursor = start + n_frames / fps


def load(meta: dict) -> VideoAvatar:
    v = meta["video"]
    up = storage.subdir("uploads")
    return VideoAvatar(up / v["frames"], up / v["track"], meta)
