"""Face analysis for uploaded avatar photos: landmarks (Haar cascades + proportions) and background removal (GrabCut)."""
import math

import cv2
import numpy as np
from PIL import Image

from . import landmarks, segment

_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def analyze(rgb: np.ndarray, near=None) -> dict:
    """rgb: HxWx3 uint8. Returns pixel-space landmarks: {found, source, face:(x,y,w,h), eyes:[(x,y,r)]*2, mouth:(cx,cy,w),
    chin, skin, roll}. Uses the MediaPipe Face Mesh models when available ("landmarks"), else Haar cascades + proportions."""
    lm = landmarks.detect_landmarks(rgb, near)
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
                "mouth": sm["mouth"], "chin": chin_y, "chin_pt": sm["chin"], "skin": skin, "roll": sm["roll"], "face_w": fw,
                "already_open": sm["already_open"], "yaw": sm["yaw"],
                "upper_inner": sm["upper_inner"], "lower_inner": sm["lower_inner"]}
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


def _grabcut_alpha(rgb: np.ndarray, info: dict) -> np.ndarray | None:
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


def cutout_alpha(rgb: np.ndarray, info: dict) -> np.ndarray | None:
    """Person mask (uint8 0..255). Prefers the person-segmentation model (keeps shoulders/clothes, removes busy
    backgrounds and other people), sharpened with GrabCut; falls back to face-seeded GrabCut alone."""
    prob = segment.person_probability(rgb)
    if prob is not None:
        alpha = _segmentation_alpha(rgb, info, prob)
        if alpha is not None:
            return alpha
    return _grabcut_alpha(rgb, info)


def _segmentation_alpha(rgb: np.ndarray, info: dict, prob: np.ndarray) -> np.ndarray | None:
    h, w = rgb.shape[:2]
    fw, chin = info["face_w"], info["chin"]
    ex = (info["eyes"][0][0] + info["eyes"][1][0]) / 2
    ey = (info["eyes"][0][1] + info["eyes"][1][1]) / 2
    body = (prob > 0.5).astype(np.uint8)
    n, lab, _, _ = cv2.connectedComponentsWithStats(body)
    fl = lab[int(np.clip((ey + chin) / 2, 0, h - 1)), int(np.clip(ex, 0, w - 1))]
    if fl == 0:
        return None
    person = (lab == fl).astype(np.uint8)  # only the person whose face we track, not people behind them
    k = max(3, int(min(h, w) * 0.012)) | 1
    near = cv2.dilate(person, np.ones((k * 3, k * 3), np.uint8))
    # sharpen the 256px model edges with GrabCut, constrained to a band around the model's outline
    gc = np.full((h, w), cv2.GC_BGD, np.uint8)
    gc[near > 0] = cv2.GC_PR_BGD
    gc[(person > 0)] = cv2.GC_PR_FGD
    gc[cv2.erode(person, np.ones((k * 2, k * 2), np.uint8)) & (prob > 0.9)] = cv2.GC_FGD
    cv2.setRNGSeed(7)
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
        refined = (((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)) & (near > 0)).astype(np.uint8)
    except cv2.error:
        refined = person
    alpha = _validate_mask(np.where(refined > 0, cv2.GC_FGD, cv2.GC_BGD).astype(np.uint8), ex, ey, chin, fw, h, w)
    if alpha is None:  # GrabCut disagreed badly: trust the model's own soft mask
        soft = (np.clip((prob - 0.3) / 0.4, 0, 1) * person).astype(np.float32)
        if soft[int(max(0, ey)): int(min(h, chin)), int(max(0, ex - 0.3 * fw)): int(min(w, ex + 0.3 * fw))].mean() < 0.8:
            return None
        return cv2.GaussianBlur((soft * 255).astype(np.uint8), (0, 0), 1.2)
    return alpha


def close_mouth(rgb: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """For photos where the mouth is open (smiling with teeth), warp the upper and lower inner lips together so the
    animation starts from a neutral closed mouth. Returns (new image, new seam y at the mouth centre)."""
    h, w = rgb.shape[:2]
    xa, xb = float(min(upper[0, 0], upper[-1, 0])), float(max(upper[0, 0], upper[-1, 0]))
    if xb - xa < 4:
        return rgb, float((upper[:, 1].mean() + lower[:, 1].mean()) / 2), seam_curve(upper, lower)
    ou, ol = np.argsort(upper[:, 0]), np.argsort(lower[:, 0])
    xs = np.arange(int(xa), int(np.ceil(xb)) + 1)
    u = np.interp(xs, upper[ou, 0], upper[ou, 1])
    lo = np.interp(xs, lower[ol, 0], lower[ol, 1])
    gap = np.clip(lo - u, 0, None)
    seam = (u + lo) / 2
    grow = np.where(gap > 0.5, np.maximum(1.5, gap * 0.12), 0)  # overshoot slightly so no line of teeth survives at the seam
    u, lo = u - grow, lo + grow
    gap = np.clip(lo - u, 0, None)
    R = float(max(4.0, gap.max() * 1.1))  # lip thickness band that gets stretched over the gap
    y0, y1 = int(max(0, u.min() - R - 2)), int(min(h, lo.max() + R + 2))
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    ys = np.arange(y0, y1, dtype=np.float32)[:, None]
    uu, ll, ss = u[None, :], lo[None, :], seam[None, :]
    top = uu - R
    src = ys.repeat(len(xs), 1)
    a = (ys >= top) & (ys <= ss)  # upper lip stretched down to the seam
    src = np.where(a, top + (ys - top) * (R / np.maximum(ss - top, 1e-3)), src)
    b = (ys > ss) & (ys <= ll + R)  # lower lip stretched up to the seam
    src = np.where(b, ll + (ys - ss) * (R / np.maximum(ll + R - ss, 1e-3)), src)
    # fade the effect out toward the mouth corners so the edges stay seamless
    t = np.clip(np.minimum(xs - xa, xb - xs) / max(2.0, (xb - xa) * 0.12), 0, 1)[None, :]
    src = ys + (src - ys) * t
    map_y[y0:y1, xs[0]: xs[-1] + 1] = src.astype(np.float32)
    map_y[y0:y1] = cv2.GaussianBlur(map_y[y0:y1], (0, 0), sigmaX=1.5, sigmaY=0.01)
    out = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    cx = (xa + xb) / 2
    curve = np.stack([np.linspace(xa, xb, 11), np.interp(np.linspace(xa, xb, 11), xs, seam)], axis=1)
    return out, float(np.interp(cx, xs, seam)), curve


def seam_curve(upper: np.ndarray, lower: np.ndarray, n: int = 11) -> np.ndarray:
    """Lip-seam polyline (n points, corner to corner): midway between the upper and lower inner-lip contours."""
    ou, ol = np.argsort(upper[:, 0]), np.argsort(lower[:, 0])
    xa, xb = float(upper[:, 0].min()), float(upper[:, 0].max())
    xs = np.linspace(xa, xb, n)
    return np.stack([xs, (np.interp(xs, upper[ou, 0], upper[ou, 1]) + np.interp(xs, lower[ol, 0], lower[ol, 1])) / 2], axis=1)


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
    if info["source"] != "landmarks" or abs(roll) < 2.5 or abs(roll) > 80:
        return rgb, info
    h, w = rgb.shape[:2]
    ex = (info["eyes"][0][0] + info["eyes"][1][0]) / 2
    ey = (info["eyes"][0][1] + info["eyes"][1][1]) / 2
    M = cv2.getRotationMatrix2D((ex, ey), roll, 1.0)  # positive angle rotates counter-clockwise: undoes a clockwise tilt
    rot = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    return rot, _transform_info(info, M)


def _transform_info(info: dict, M: np.ndarray) -> dict:
    """Apply the 2x3 affine M (the rotation used on the image) to every landmark, so no second detection is needed."""
    tp = lambda x, y: (float(M[0, 0] * x + M[0, 1] * y + M[0, 2]), float(M[1, 0] * x + M[1, 1] * y + M[1, 2]))
    out = dict(info)
    out["eyes"] = sorted([(*tp(x, y), r) for x, y, r in info["eyes"]], key=lambda e: e[0])
    mx, my, mw = info["mouth"]
    out["mouth"] = (*tp(mx, my), mw)
    cx, cy = tp(*info.get("chin_pt", (mx, info["chin"])))
    out["chin"], out["chin_pt"] = cy, (cx, cy)
    for k in ("upper_inner", "lower_inner"):
        if k in info:
            pts = np.asarray(info[k], np.float32)
            out[k] = pts @ M[:, :2].T.astype(np.float32) + M[:, 2].astype(np.float32)
    fw = info["face_w"]
    ex = (out["eyes"][0][0] + out["eyes"][1][0]) / 2
    out["face"] = (ex - fw / 2, cy - fw, fw, fw)
    out["roll"] = 0.0
    return out


def _shift_info(info: dict, dx: float, dy: float) -> dict:
    """Translate every pixel-space landmark in an analyze() result (used after analysing a crop)."""
    out = dict(info)
    fx, fy, fw, fh = info["face"]
    out["face"] = (fx + dx, fy + dy, fw, fh)
    out["eyes"] = [(x + dx, y + dy, r) for x, y, r in info["eyes"]]
    mx, my, mw = info["mouth"]
    out["mouth"] = (mx + dx, my + dy, mw)
    out["chin"] = info["chin"] + dy
    for k in ("upper_inner", "lower_inner"):
        if k in info:
            out[k] = info[k] + np.array([dx, dy], np.float32)
    return out


def build_avatar(img: Image.Image, remove_bg: bool, gender: str) -> tuple[Image.Image, dict]:
    """Photo -> square RGBA avatar canvas (face normalised to the same layout as the bundled avatars) + landmark metadata."""
    img = img.convert("RGB")
    img.thumbnail((1024, 1024))
    rgb = np.asarray(img)
    info = analyze(rgb)
    roll_before = info.get("roll", 0.0)
    rgb, info = _upright(rgb, info)
    info["roll_before"] = roll_before
    if info.get("already_open") and "upper_inner" in info:
        rgb, seam, curve = close_mouth(rgb, info["upper_inner"], info["lower_inner"])
        mx, _, mw = info["mouth"]
        info["mouth"] = (mx, seam, mw)
        info["seam"] = curve
        info["already_open"] = False
        info["mouth_closed_by_warp"] = True
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
        "closed_by_warp": bool(info.get("mouth_closed_by_warp")),
    }
    if "seam" not in info and "upper_inner" in info:
        info["seam"] = seam_curve(info["upper_inner"], info["lower_inner"])
    if "seam" in info:
        meta["seam"] = [[tx(x), ty(y)] for x, y in info["seam"]]
        sx = [p[0] for p in meta["seam"]]
        sy = [p[1] for p in meta["seam"]]
        cxs = (min(sx) + max(sx)) / 2  # centre the animated opening on the inner lip line itself
        meta["mouth"] = [cxs, float(np.interp(cxs, sx, sy)), max(meta["mouth"][2], (max(sx) - min(sx)) * 1.06)]
    warn = []
    if not info["found"]:
        warn.append("No face was detected, so the mouth and eye positions are guesses. Try a clear, front-facing photo.")
    elif abs(info.get("roll_before", 0)) > 35 or abs(info.get("yaw", 0)) > 0.3:
        warn.append("The head is strongly tilted or turned. The animation works best with a front-facing photo.")
    if warn:
        meta["warning"] = " ".join(warn)
    return canvas, meta
