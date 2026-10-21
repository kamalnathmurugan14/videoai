"""Facial landmarks via Google's MediaPipe BlazeFace detector + 478-point Face Mesh models (Apache-2.0 TFLite files in
assets/models), run with ai-edge-litert. Gives exact lip, eye, iris and jaw positions for any face pose in a photo.
Everything degrades gracefully: if the runtime or models are missing, `detect_landmarks` returns None and callers fall back
to the Haar-cascade heuristics in face.py."""
import logging
import math
import threading
from functools import lru_cache

import cv2
import numpy as np

from ..config import ASSETS_DIR

log = logging.getLogger(__name__)
_MODELS = ASSETS_DIR / "models"
_lock = threading.Lock()

# FaceMesh landmark indices (MediaPipe canonical face model)
LIP_UPPER_INNER, LIP_LOWER_INNER = 13, 14
MOUTH_CORNER_A, MOUTH_CORNER_B = 61, 291
LIP_TOP_OUTER, LIP_BOTTOM_OUTER = 0, 17
CHIN, FOREHEAD = 152, 10
FACE_SIDE_A, FACE_SIDE_B = 234, 454
EYE_A = (33, 133, 159, 145, 468)   # outer corner, inner corner, upper lid, lower lid, iris centre
EYE_B = (263, 362, 386, 374, 473)
CHEEK_A, CHEEK_B = 205, 425
UPPER_INNER = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
LOWER_INNER = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)


@lru_cache(maxsize=1)
def _interpreters():
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception as e:  # runtime not installed
        log.warning("face landmarks disabled (ai-edge-litert unavailable: %s)", e)
        return None
    try:
        det = Interpreter(model_path=str(_MODELS / "face_detector.tflite"))
        mesh = Interpreter(model_path=str(_MODELS / "face_landmarks_detector.tflite"))
        det.allocate_tensors()
        mesh.allocate_tensors()
        return det, mesh
    except Exception as e:
        log.warning("face landmark models failed to load: %s", e)
        return None


@lru_cache(maxsize=1)
def _anchors() -> np.ndarray:
    """BlazeFace short-range SSD anchors (896): strides 8,16,16,16 on a 128x128 input, 2 anchors per cell for the
    stride-8 layer and 6 for the merged stride-16 layers; fixed anchor size."""
    out = []
    for fm, per in ((16, 2), (8, 6)):
        for y in range(fm):
            for x in range(fm):
                for _ in range(per):
                    out.append(((x + 0.5) / fm, (y + 0.5) / fm))
    return np.array(out, np.float32)


def _letterbox(rgb: np.ndarray, size: int):
    h, w = rgb.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    pad = np.zeros((size, size, 3), np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    pad[oy:oy + nh, ox:ox + nw] = img
    return pad, s, ox, oy


def _detect_faces(rgb: np.ndarray, det) -> list[dict]:
    pad, s, ox, oy = _letterbox(rgb, 128)
    x = (pad.astype(np.float32) / 127.5 - 1.0)[None]
    inp = det.get_input_details()[0]
    det.set_tensor(inp["index"], x)
    det.invoke()
    outs = {d["name"]: det.get_tensor(d["index"]) for d in det.get_output_details()}
    raw = next(v for k, v in outs.items() if v.shape[-1] == 16)[0]
    score = 1 / (1 + np.exp(-np.clip(next(v for k, v in outs.items() if v.shape[-1] == 1)[0, :, 0], -30, 30)))
    a = _anchors()
    keep = np.where(score > 0.5)[0]
    cands = []
    for i in keep:
        xc, yc = raw[i, 0] / 128 + a[i, 0], raw[i, 1] / 128 + a[i, 1]
        w, h = raw[i, 2] / 128, raw[i, 3] / 128
        kps = [(raw[i, 4 + 2 * k] / 128 + a[i, 0], raw[i, 5 + 2 * k] / 128 + a[i, 1]) for k in range(6)]
        # letterboxed-unit coords -> source pixels
        to_px = lambda u, v: ((u * 128 - ox) / s, (v * 128 - oy) / s)
        cx, cy = to_px(xc, yc)
        cands.append({"score": float(score[i]), "cx": cx, "cy": cy, "w": w * 128 / s, "h": h * 128 / s,
                      "kps": [to_px(*k) for k in kps]})
    # greedy NMS on IoU
    cands.sort(key=lambda c: -c["score"])
    kept = []
    for c in cands:
        ok = True
        for k in kept:
            ix = max(0, min(c["cx"] + c["w"] / 2, k["cx"] + k["w"] / 2) - max(c["cx"] - c["w"] / 2, k["cx"] - k["w"] / 2))
            iy = max(0, min(c["cy"] + c["h"] / 2, k["cy"] + k["h"] / 2) - max(c["cy"] - c["h"] / 2, k["cy"] - k["h"] / 2))
            inter = ix * iy
            if inter / (c["w"] * c["h"] + k["w"] * k["h"] - inter + 1e-6) > 0.3:
                ok = False
                break
        if ok:
            kept.append(c)
    return kept


def _mesh(rgb: np.ndarray, face: dict, mesh) -> np.ndarray | None:
    """Run Face Mesh on a rotation-aligned crop around a detection. Returns 478x3 landmarks in source pixels (z unused)."""
    (rx, ry), (lx, ly) = face["kps"][0], face["kps"][1]  # the two eyes
    theta = math.atan2(ly - ry, lx - rx)  # eye line angle; rotate the crop so it becomes horizontal
    size = max(face["w"], face["h"]) * 1.5
    c, sn = math.cos(theta), math.sin(theta)
    sc = 256 / size
    # affine: source -> 256x256 crop (rotate by -theta about the box centre, scale, centre)
    M = np.array([[c * sc, sn * sc, 0], [-sn * sc, c * sc, 0]], np.float32)
    M[:, 2] = 128 - M[:, :2] @ np.array([face["cx"], face["cy"]], np.float32)
    crop = cv2.warpAffine(rgb, M, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    inp = mesh.get_input_details()[0]
    mesh.set_tensor(inp["index"], (crop.astype(np.float32) / 255.0)[None])
    mesh.invoke()
    outs = {d["name"]: mesh.get_tensor(d["index"]) for d in mesh.get_output_details()}
    pts = next(v for v in outs.values() if v.size == 478 * 3).reshape(478, 3)
    flag = next((float(v.ravel()[0]) for v in outs.values() if v.size == 1), 1.0)
    if 1 / (1 + math.exp(-max(-50, min(50, flag)))) < 0.5:
        return None
    inv = cv2.invertAffineTransform(M)
    xy = pts[:, :2] @ inv[:, :2].T + inv[:, 2]
    return xy.astype(np.float32)


def _best_face(rgb: np.ndarray, det, mesh, near=None):
    """Largest face with landmarks in `rgb` (whole image only); with `near`=(x, y), the face closest to that point."""
    faces = _detect_faces(rgb, det)
    key = (lambda f: -f["w"] * f["h"]) if near is None else (lambda f: (f["cx"] - near[0]) ** 2 + (f["cy"] - near[1]) ** 2)
    for face in sorted(faces, key=key):
        pts = _mesh(rgb, face, mesh)
        if pts is not None:
            return face, pts
    return None


def _in_crop(rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int, det, mesh, scale_up: float = 1.0):
    crop = rgb[y0:y1, x0:x1]
    if scale_up != 1.0:
        crop = cv2.resize(crop, None, fx=scale_up, fy=scale_up, interpolation=cv2.INTER_CUBIC)
    r = _best_face(crop, det, mesh)
    if r is None:
        return None
    face, pts = r
    pts = pts / scale_up + np.array([x0, y0], np.float32)
    face = {**face, "cx": face["cx"] / scale_up + x0, "cy": face["cy"] / scale_up + y0, "w": face["w"] / scale_up,
            "h": face["h"] / scale_up}
    return face, pts


def detect_landmarks(rgb: np.ndarray, near=None) -> dict | None:
    """Landmarks of the largest face: dict with 'pts' (478x2 source px) and 'score', or None if no face / unavailable.
    Tries the whole image first, then overlapping tiles, then zooms into faces found by a Haar cascade (small faces)."""
    models = _interpreters()
    if models is None:
        return None
    det, mesh = models
    h, w = rgb.shape[:2]
    with _lock:
        try:
            found = _best_face(rgb, det, mesh, near)
            if found is None:  # overlapping tiles at two sizes
                best = None
                for frac, n in ((0.6, 2), (0.4, 3)):
                    tw, th = int(w * frac), int(h * frac)
                    for iy in range(n):
                        for ix in range(n):
                            x0 = int((w - tw) * ix / max(1, n - 1))
                            y0 = int((h - th) * iy / max(1, n - 1))
                            r = _in_crop(rgb, x0, y0, x0 + tw, y0 + th, det, mesh)
                            if r and (best is None or r[0]["w"] * r[0]["h"] > best[0]["w"] * best[0]["h"]):
                                best = r
                    if best:
                        break
                found = best
            if found is None:  # tiny faces: locate with Haar, then zoom in around the box
                gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
                cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                boxes = cascade.detectMultiScale(gray, 1.08, 4, minSize=(max(20, int(min(h, w) * 0.025)),) * 2)
                for (fx, fy, fw, fh) in sorted(boxes, key=lambda b: -b[2] * b[3]):
                    pad = int(fw * 1.2)
                    x0, y0, x1, y1 = max(0, fx - pad), max(0, fy - pad), min(w, fx + fw + pad), min(h, fy + fh + pad)
                    found = _in_crop(rgb, x0, y0, x1, y1, det, mesh, scale_up=max(1.0, 256 / (x1 - x0)))
                    if found:
                        break
        except Exception as e:
            log.warning("landmark detection failed: %s", e)
            return None
    if found is None:
        return None
    face, pts = found
    return {"pts": pts.astype(np.float32), "score": face["score"], "box": (face["cx"], face["cy"], face["w"], face["h"])}


def track_landmarks(rgb: np.ndarray, prev_pts: np.ndarray) -> dict | None:
    """Video tracking: run Face Mesh on the region where the face was in the previous frame (no detector pass, so it is
    fast and does not jump between faces). Returns the same dict as detect_landmarks, or None if the face was lost."""
    models = _interpreters()
    if models is None:
        return None
    _, mesh = models
    (x0, y0), (x1, y1) = prev_pts.min(0), prev_pts.max(0)
    eyes = sorted([tuple(prev_pts[468]), tuple(prev_pts[473])])
    side = max(x1 - x0, y1 - y0) * 0.96  # ~ the BlazeFace box size for this face
    face = {"cx": float(x0 + x1) / 2, "cy": float(y0 + y1) / 2 + side * 0.02, "w": side, "h": side, "kps": eyes, "score": 1.0}
    with _lock:
        try:
            pts = _mesh(rgb, face, mesh)
        except Exception as e:
            log.warning("landmark tracking failed: %s", e)
            return None
    if pts is None:
        return None
    return {"pts": pts.astype(np.float32), "score": 1.0, "box": (face["cx"], face["cy"], side, side)}


def summarize(lm: dict) -> dict:
    """Reduce 478 landmarks to what the animator needs (pixel space)."""
    p = lm["pts"]
    pt = lambda i: p[i]
    ua, la = pt(LIP_UPPER_INNER), pt(LIP_LOWER_INNER)
    ca, cb = pt(MOUTH_CORNER_A), pt(MOUTH_CORNER_B)
    mouth_w = float(np.hypot(*(cb - ca)))
    gap = float(abs(la[1] - ua[1]))
    already_open = gap / max(mouth_w, 1e-6) > 0.07  # smiling / talking in the photo: teeth or cavity already visible
    seam = la[1] if already_open else (ua[1] + la[1]) / 2  # open: animate from the lower inner lip so the photo's own opening stays
    mouth = ((ca[0] + cb[0]) / 2, float(seam), mouth_w)
    eyes = []
    for ids in (EYE_A, EYE_B):
        outer, inner, up, lo, iris = (pt(i) for i in ids)
        eyes.append((float(iris[0]), float((up[1] + lo[1]) / 2), float(np.hypot(*(outer - inner)) * 0.5)))
    eyes.sort(key=lambda e: e[0])
    side_a, side_b = pt(FACE_SIDE_A), pt(FACE_SIDE_B)
    face_w = float(np.hypot(*(side_b - side_a)))
    chin = pt(CHIN)
    top = pt(FOREHEAD)
    roll = math.degrees(math.atan2(eyes[1][1] - eyes[0][1], eyes[1][0] - eyes[0][0]))
    eye_dist = max(1e-6, float(np.hypot(eyes[1][0] - eyes[0][0], eyes[1][1] - eyes[0][1])))
    nose = pt(1)
    yaw = float((nose[0] - (eyes[0][0] + eyes[1][0]) / 2) / eye_dist)  # ~0 for a frontal face, grows as the head turns
    return {"upper_inner": p[list(UPPER_INNER)].copy(), "lower_inner": p[list(LOWER_INNER)].copy(),
            "already_open": already_open, "yaw": yaw, "mouth": mouth, "eyes": eyes, "face_w": face_w, "chin": (float(chin[0]), float(chin[1])),
            "top": (float(top[0]), float(top[1])), "roll": roll, "cheeks": (pt(CHEEK_A), pt(CHEEK_B))}
