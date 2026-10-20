"""Face analysis for uploaded avatar photos: landmarks (Haar cascades + proportions) and background removal (GrabCut)."""
import math

import cv2
import numpy as np
from PIL import Image

from . import landmarks

_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def analyze(rgb: np.ndarray) -> dict:
    """rgb: HxWx3 uint8. Returns pixel-space landmarks: {found, source, face:(x,y,w,h), eyes:[(x,y,r)]*2, mouth:(cx,cy,w),
    chin, skin, roll}. Uses the MediaPipe Face Mesh models when available ("landmarks"), else Haar cascades + proportions."""
    lm = landmarks.detect_landmarks(rgb)
    if lm is not None:
        sm = landmarks.summarize(lm)
        fw = sm["face_w"]
        chin_y = sm["chin"][1]
        cheek = np.mean(sm["cheeks"], axis=0)
        r = max(2, int(fw * 0.05))
        x0, y0 = int(np.clip(cheek[0] - r, 0, rgb.shape[1] - 2)), int(np.clip(cheek[1] - r, 0, rgb.shape[0] - 2))
        patch = rgb[y0: y0 + 2 * r, x0: x0 + 2 * r]
        skin = tuple(int(v) for v in np.median(patch.reshape(-1, 3), axis=0)) if patch.size else (224, 172, 138)
        cx = (sm["eyes"][0][0] + sm["eyes"][1][0]) / 2
        return {"found": True, "source": "landmarks", "face": (cx - fw / 2, chin_y - fw, fw, fw), "eyes": sm["eyes"],
                "mouth": sm["mouth"], "chin": chin_y, "skin": skin, "roll": sm["roll"], "face_w": fw,
                "already_open": sm["already_open"], "yaw": sm["yaw"]}
    info = _analyze_haar(rgb)
    info["source"] = "haar" if info["found"] else "guess"
    info["roll"] = 0.0
    info["already_open"] = False
    info["yaw"] = 0.0
    info["face_w"] = info["face"][2] * 0.84  # comparable to the mesh's cheek-to-cheek width
    return info


def _analyze_haar(rgb: np.ndarray) -> dict:
    h, w = rgb.shape[:2]
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    min_side = int(min(h, w) * 0.12)
    faces = _face.detectMultiScale(gray, 1.1, 5, minSize=(min_side, min_side))
    if len(faces) == 0:
        faces = _face.detectMultiScale(gray, 1.05, 3, minSize=(min_side, min_side))
    if len(faces) == 0:  # no face: assume a centred head-and-shoulders portrait
        fw = w * 0.45
        fx, fy = w * 0.275, h * 0.18
        found, fh = False, fw
    else:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        found = True
    fx, fy, fw, fh = float(fx), float(fy), float(fw), float(fh)

    eyes = []
    if found:
        roi = gray[int(fy): int(fy + fh * 0.62), int(fx): int(fx + fw)]
        cand = _eye.detectMultiScale(roi, 1.1, 4, minSize=(int(fw * 0.12), int(fw * 0.12)))
        left = [e for e in cand if e[0] + e[2] / 2 < fw * 0.5 and e[1] + e[3] / 2 < fh * 0.5]
        right = [e for e in cand if e[0] + e[2] / 2 >= fw * 0.5 and e[1] + e[3] / 2 < fh * 0.5]
        if left and right:
            for grp in (left, right):
                ex, ey, ew, eh = max(grp, key=lambda e: e[2] * e[3])
                eyes.append((fx + ex + ew / 2, fy + ey + eh / 2, ew * 0.42))
    eyes_found = len(eyes) == 2
    if not eyes_found:
        r = fw * 0.095
        eyes = [(fx + fw * 0.31, fy + fh * 0.40, r), (fx + fw * 0.69, fy + fh * 0.40, r)]

    mid_x = (eyes[0][0] + eyes[1][0]) / 2
    mid_y = (eyes[0][1] + eyes[1][1]) / 2
    dist = abs(eyes[1][0] - eyes[0][0])
    mouth_cy = fy + fh * 0.80  # the Haar box runs brow->chin, so the lip seam sits ~80% down it
    mouth = (mid_x, mouth_cy, fw * 0.32)
    chin = fy + fh * 1.04

    px = int(np.clip(mid_x - fw * 0.22, 0, w - 2)), int(np.clip(mid_y + fh * 0.22, 0, h - 2))
    patch = rgb[px[1]: px[1] + max(2, int(fh * 0.1)), px[0]: px[0] + max(2, int(fw * 0.12))]
    skin = tuple(int(v) for v in np.median(patch.reshape(-1, 3), axis=0)) if patch.size else (224, 172, 138)
    return {"found": found, "face": (fx, fy, fw, fh), "eyes": eyes, "mouth": mouth, "chin": chin, "skin": skin}


def cutout_alpha(rgb: np.ndarray, info: dict) -> np.ndarray | None:
    """Person mask (uint8 0..255) via GrabCut seeded from the face geometry; None if the result looks implausible
    (so the caller keeps the original photo background instead of producing a broken avatar)."""
    h, w = rgb.shape[:2]
    fw, chin = info["face_w"], info["chin"]
    ex = (info["eyes"][0][0] + info["eyes"][1][0]) / 2
    ey = (info["eyes"][0][1] + info["eyes"][1][1]) / 2
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    x0, x1 = int(max(0, ex - 1.9 * fw)), int(min(w, ex + 1.9 * fw))
    y0 = int(max(0, ey - 1.8 * fw))
    mask[y0:h, x0:x1] = cv2.GC_PR_BGD
    poly = np.array([
        (ex - 0.62 * fw, ey - 1.15 * fw), (ex + 0.62 * fw, ey - 1.15 * fw), (ex + 0.78 * fw, ey - 0.3 * fw),
        (ex + 0.8 * fw, chin), (ex + 1.5 * fw, chin + 0.45 * fw), (ex + 1.7 * fw, h + 5),
        (ex - 1.7 * fw, h + 5), (ex - 1.5 * fw, chin + 0.45 * fw), (ex - 0.8 * fw, chin), (ex - 0.78 * fw, ey - 0.3 * fw)], np.int32)
    cv2.fillPoly(mask, [poly], cv2.GC_PR_FGD)
    cv2.ellipse(mask, (int(ex), int((ey + chin) / 2)), (int(0.34 * fw), int(0.55 * (chin - ey) + 0.05 * fw)), 0, 0, 360, cv2.GC_FGD, -1)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for seed in (7, 1234, 99):  # GrabCut's k-means init is random: seed it so results are repeatable, retry if rejected
        cv2.setRNGSeed(seed)
        m = mask.copy()
        bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
        try:
            cv2.grabCut(bgr, m, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return None
        alpha = _validate_mask(m, ex, ey, chin, fw, h, w)
        if alpha is not None:
            return alpha
    return None


def _validate_mask(m: np.ndarray, ex: float, ey: float, chin: float, fw: float, h: int, w: int) -> np.ndarray | None:
    fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return None
    face_label = lab[int(np.clip((ey + chin) / 2, 0, h - 1)), int(np.clip(ex, 0, w - 1))]
    if face_label == 0:  # the face itself must belong to the person mask
        return None
    fg = (lab == face_label).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    face_box = fg[int(max(0, ey - 0.3 * fw)): int(min(h, chin)), int(max(0, ex - 0.35 * fw)): int(min(w, ex + 0.35 * fw))]
    if face_box.size == 0 or face_box.mean() < 0.9 or not 0.08 < fg.mean() < 0.85:
        return None
    return cv2.GaussianBlur(fg * 255, (0, 0), 1.6)


CANVAS = 768


def _upright(rgb: np.ndarray, info: dict) -> tuple[np.ndarray, dict]:
    """Rotate a tilted photo so the eyes are level (the puppet animation assumes an upright face), then re-detect."""
    roll = info.get("roll", 0.0)
    if info["source"] != "landmarks" or abs(roll) < 2.5 or abs(roll) > 60:
        return rgb, info
    h, w = rgb.shape[:2]
    ex = (info["eyes"][0][0] + info["eyes"][1][0]) / 2
    ey = (info["eyes"][0][1] + info["eyes"][1][1]) / 2
    M = cv2.getRotationMatrix2D((ex, ey), roll, 1.0)  # positive angle rotates counter-clockwise: undoes a clockwise tilt
    rot = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    again = analyze(rot)
    return (rot, again) if again["source"] == "landmarks" else (rgb, info)


def build_avatar(img: Image.Image, remove_bg: bool, gender: str) -> tuple[Image.Image, dict]:
    """Photo -> square RGBA avatar canvas (face normalised to the same layout as the bundled avatars) + landmark metadata."""
    img = img.convert("RGB")
    img.thumbnail((1024, 1024))
    rgb = np.asarray(img)
    info = analyze(rgb)
    roll_before = info.get("roll", 0.0)
    rgb, info = _upright(rgb, info)
    info["roll_before"] = roll_before
    h, w = rgb.shape[:2]
    alpha = cutout_alpha(rgb, info) if (remove_bg and info["found"]) else None
    cut = alpha is not None
    a = alpha.astype(np.float32) / 255 if cut else np.ones((h, w), np.float32)
    # soften the photo's own edges so a cropped bust dissolves into the scene
    yy, xx = np.mgrid[0:h, 0:w]
    ramp = lambda d, n: np.clip(d / max(1, n), 0, 1)
    a *= ramp(h - 1 - yy, h * 0.14) * ramp(xx, w * 0.05) * ramp(w - 1 - xx, w * 0.05)
    if not cut:
        a *= ramp(yy, h * 0.04)
    rgba = Image.fromarray(np.dstack([rgb, (a * 255).astype(np.uint8)]), "RGBA")

    eyes = info["eyes"]
    ecx, ecy = (eyes[0][0] + eyes[1][0]) / 2, (eyes[0][1] + eyes[1][1]) / 2
    scale = (0.39 * CANVAS) / info["face_w"]
    ox, oy = 0.5 * CANVAS - ecx * scale, 0.40 * CANVAS - ecy * scale
    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    resized = rgba.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    canvas.paste(resized, (round(ox), round(oy)), resized)

    tx = lambda x: float(np.clip((x * scale + ox) / CANVAS, 0, 1))
    ty = lambda y: float(np.clip((y * scale + oy) / CANVAS, 0, 1))
    mx, my, mw = info["mouth"]
    meta = {
        "found": info["found"], "source": info["source"], "cutout": cut, "gender": gender, "has_mouth_art": False,
        "mouth": [tx(mx), ty(my), mw * scale / CANVAS],
        "eyes": [[tx(x), ty(y), r * scale / CANVAS] for x, y, r in eyes],
        "chin": ty(info["chin"]), "face_w": 0.46,  # canvas is normalised so the animation constants are tuned for 0.46
        "skin": "#%02x%02x%02x" % info["skin"], "open": bool(info.get("already_open")),
    }
    warn = []
    if not info["found"]:
        warn.append("No face was detected, so the mouth and eye positions are guesses. Try a clear, front-facing photo.")
    elif abs(info.get("roll_before", 0)) > 35 or abs(info.get("yaw", 0)) > 0.3:
        warn.append("The head is strongly tilted or turned. The animation works best with a front-facing photo.")
    if warn:
        meta["warning"] = " ".join(warn)
    return canvas, meta
