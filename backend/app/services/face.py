"""Face analysis for uploaded avatar photos: landmarks (Haar cascades + proportions) and background removal (GrabCut)."""
import cv2
import numpy as np
from PIL import Image

_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def analyze(rgb: np.ndarray) -> dict:
    """rgb: HxWx3 uint8. Returns pixel-space landmarks: {found, face:(x,y,w,h), eyes:[(x,y,r)]*2, mouth:(cx,cy,w), chin, skin}."""
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
    mouth = (mid_x, mouth_cy, fw * 0.42)
    chin = fy + fh * 1.04

    px = int(np.clip(mid_x - fw * 0.22, 0, w - 2)), int(np.clip(mid_y + fh * 0.22, 0, h - 2))
    patch = rgb[px[1]: px[1] + max(2, int(fh * 0.1)), px[0]: px[0] + max(2, int(fw * 0.12))]
    skin = tuple(int(v) for v in np.median(patch.reshape(-1, 3), axis=0)) if patch.size else (224, 172, 138)
    return {"found": found, "face": (fx, fy, fw, fh), "eyes": eyes, "mouth": mouth, "chin": chin, "skin": skin}


def cutout_alpha(rgb: np.ndarray, face: tuple[float, float, float, float]) -> np.ndarray | None:
    """GrabCut person mask (uint8 0..255) seeded from the face box; None if the result looks implausible."""
    h, w = rgb.shape[:2]
    fx, fy, fw, fh = face
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    x0, x1 = int(max(0, fx - fw * 0.55)), int(min(w, fx + fw + fw * 0.55))
    mask[int(max(0, fy - fh * 0.55)):h, x0:x1] = cv2.GC_PR_BGD
    poly = np.array([[fx - fw * 0.15, fy - fh * 0.35], [fx + fw * 1.15, fy - fh * 0.35],
                     [fx + fw * 1.25, h], [fx - fw * 0.25, h]], np.int32)
    cv2.fillPoly(mask, [poly], cv2.GC_PR_FGD)
    cv2.ellipse(mask, (int(fx + fw / 2), int(fy + fh / 2)), (int(fw * 0.38), int(fh * 0.5)), 0, 0, 360, cv2.GC_FGD, -1)
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return None
    fg = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if not 0.12 < fg.mean() < 0.85:
        return None
    return cv2.GaussianBlur(fg * 255, (0, 0), 1.6)


CANVAS = 768


def build_avatar(img: Image.Image, remove_bg: bool, gender: str) -> tuple[Image.Image, dict]:
    """Photo -> square RGBA avatar canvas (face normalised to the same layout as the bundled avatars) + landmark metadata."""
    img = img.convert("RGB")
    img.thumbnail((1024, 1024))
    rgb = np.asarray(img)
    h, w = rgb.shape[:2]
    info = analyze(rgb)
    alpha = cutout_alpha(rgb, info["face"]) if (remove_bg and info["found"]) else None
    cut = alpha is not None
    a = alpha.astype(np.float32) / 255 if cut else np.ones((h, w), np.float32)
    # soften the photo's own edges so a cropped bust dissolves into the scene
    yy, xx = np.mgrid[0:h, 0:w]
    ramp = lambda d, n: np.clip(d / max(1, n), 0, 1)
    a *= ramp(h - 1 - yy, h * 0.14) * ramp(xx, w * 0.05) * ramp(w - 1 - xx, w * 0.05)
    if not cut:
        a *= ramp(yy, h * 0.04)
    rgba = Image.fromarray(np.dstack([rgb, (a * 255).astype(np.uint8)]), "RGBA")

    fx, fy, fw, fh = info["face"]
    eyes = info["eyes"]
    ecx, ecy = (eyes[0][0] + eyes[1][0]) / 2, (eyes[0][1] + eyes[1][1]) / 2
    scale = (0.46 * CANVAS) / fw
    ox, oy = 0.5 * CANVAS - ecx * scale, 0.40 * CANVAS - ecy * scale
    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    resized = rgba.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    canvas.paste(resized, (round(ox), round(oy)), resized)

    tx = lambda x: float(np.clip((x * scale + ox) / CANVAS, 0, 1))
    ty = lambda y: float(np.clip((y * scale + oy) / CANVAS, 0, 1))
    mx, my, mw = info["mouth"]
    meta = {
        "found": info["found"], "cutout": cut, "gender": gender, "has_mouth_art": False,
        "mouth": [tx(mx), ty(my), mw * scale / CANVAS],
        "eyes": [[tx(x), ty(y), r * scale / CANVAS] for x, y, r in eyes],
        "chin": ty(info["chin"]), "face_w": fw * scale / CANVAS,
        "skin": "#%02x%02x%02x" % info["skin"],
    }
    return canvas, meta
