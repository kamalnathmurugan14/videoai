"""Generates the stylised avatar PNGs shipped in assets/avatars (run from backend/: python scripts/make_avatars.py).
Each avatar has a mouth-less image (used for animation) and a *_preview.png with a drawn smile."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.avatar import draw_mouth  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "assets" / "avatars"
S, SS = 512, 3

AVATARS = [
    dict(id="aria", gender="female", name="Aria", skin="#f1c7a5", hair="#3b2417", style="long", shirt="#6366f1", eye="#4a2f1d"),
    dict(id="leo", gender="male", name="Leo", skin="#c68a63", hair="#1c1410", style="short", shirt="#0ea5e9", eye="#2b1b17"),
    dict(id="maya", gender="female", name="Maya", skin="#8d5a3b", hair="#120d0a", style="bun", shirt="#f43f5e", eye="#1d120d"),
    dict(id="sam", gender="male", name="Sam", skin="#f6d5b8", hair="#d9a441", style="short", shirt="#10b981", eye="#3a6ea5"),
    dict(id="noor", gender="female", name="Noor", skin="#d9a273", hair="#2a1a12", style="hijab", shirt="#f59e0b", eye="#3b2417"),
    dict(id="robo", gender="male", name="Robo", skin="#cbd5e1", hair="#64748b", style="robot", shirt="#334155", eye="#22d3ee"),
]


def draw_avatar(a: dict, with_mouth: bool) -> tuple[Image.Image, dict]:
    s = S * SS
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    k = s / 512
    cx = s / 2
    skin = a["skin"]
    robot = a["style"] == "robot"
    if a["style"] == "long":
        d.rounded_rectangle([cx - 150 * k, 70 * k, cx + 150 * k, 380 * k], radius=int(110 * k), fill=a["hair"])
    # shoulders / shirt
    d.rounded_rectangle([cx - 230 * k, 400 * k, cx + 230 * k, 560 * k], radius=int(90 * k), fill=a["shirt"])
    # neck
    d.rectangle([cx - 38 * k, 330 * k, cx + 38 * k, 420 * k], fill=skin)
    d.polygon([(cx - 55 * k, 400 * k), (cx + 55 * k, 400 * k), (cx, 450 * k)], fill=skin)
    # hair back
    if a["style"] == "bun":
        d.ellipse([cx - 55 * k, 20 * k, cx + 55 * k, 100 * k], fill=a["hair"])
    if a["style"] == "hijab":
        d.rounded_rectangle([cx - 155 * k, 55 * k, cx + 155 * k, 470 * k], radius=int(130 * k), fill=a["hair"])
    # head
    if robot:
        d.rounded_rectangle([cx - 125 * k, 105 * k, cx + 125 * k, 365 * k], radius=int(55 * k), fill=skin)
        d.line([cx, 105 * k, cx, 60 * k], fill=a["hair"], width=int(8 * k))
        d.ellipse([cx - 16 * k, 40 * k, cx + 16 * k, 72 * k], fill="#f43f5e")
    else:
        d.ellipse([cx - 120 * k, 90 * k, cx + 120 * k, 360 * k], fill=skin)
        d.ellipse([cx - 128 * k, 215 * k, cx - 108 * k, 265 * k], fill=skin)  # ears
        d.ellipse([cx + 108 * k, 215 * k, cx + 128 * k, 265 * k], fill=skin)
    # hair front
    if a["style"] in ("long", "bun", "short"):
        d.pieslice([cx - 128 * k, 70 * k, cx + 128 * k, 300 * k], 180, 360, fill=a["hair"])
        d.polygon([(cx - 128 * k, 185 * k), (cx - 60 * k, 130 * k), (cx + 128 * k, 185 * k), (cx + 128 * k, 150 * k), (cx, 80 * k), (cx - 128 * k, 150 * k)], fill=a["hair"])
    if a["style"] == "hijab":
        d.pieslice([cx - 135 * k, 70 * k, cx + 135 * k, 330 * k], 180, 360, fill=a["hair"])
        d.ellipse([cx - 100 * k, 120 * k, cx + 100 * k, 360 * k], fill=skin)
    # eyes
    ey, er = 205 * k, 13 * k
    for sx in (-48, 48):
        ex = cx + sx * k
        if robot:
            d.rounded_rectangle([ex - 24 * k, ey - 18 * k, ex + 24 * k, ey + 18 * k], 10 * k, fill="#0f172a")
            d.ellipse([ex - 9 * k, ey - 9 * k, ex + 9 * k, ey + 9 * k], fill=a["eye"])
        else:
            d.ellipse([ex - 20 * k, ey - 14 * k, ex + 20 * k, ey + 14 * k], fill="white")
            d.ellipse([ex - 10 * k, ey - 10 * k, ex + 10 * k, ey + 10 * k], fill=a["eye"])
            d.ellipse([ex - 4 * k, ey - 5 * k, ex + 4 * k, ey + 3 * k], fill="#111")
            d.arc([ex - 26 * k, ey - 38 * k, ex + 26 * k, ey - 6 * k], 200, 340, fill=a["hair"] if a["style"] != "robot" else "#475569", width=int(7 * k))
    if not robot:
        d.arc([cx - 12 * k, 215 * k, cx + 12 * k, 262 * k], 90, 270, fill="#b9825f", width=int(4 * k))  # nose
        blush = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        bd = ImageDraw.Draw(blush)
        for sx in (-70, 70):
            bd.ellipse([cx + sx * k - 16 * k, 250 * k, cx + sx * k + 16 * k, 272 * k], fill=(255, 110, 110, 50))
        im = Image.alpha_composite(im, blush)
    im = im.resize((S, S), Image.LANCZOS)
    meta = {"id": a["id"], "name": a["name"], "mouth": [0.5, 300 / 512, 0.17], "skin": skin,
            "eyes": None if robot else [[(256 - 48) / 512, 205 / 512, 0.03], [(256 + 48) / 512, 205 / 512, 0.03]],
            "has_mouth_art": True, "chin": 360 / 512, "face_w": 240 / 512, "gender": a["gender"], "image": f"{a['id']}.png", "preview": f"{a['id']}_preview.png"}
    if with_mouth:
        draw_mouth(im, 256, 300, 0.17 * S, 0.0)
    return im, meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metas = []
    for a in AVATARS:
        base, meta = draw_avatar(a, False)
        prev, _ = draw_avatar(a, True)
        base.save(OUT / meta["image"], optimize=True)
        prev.save(OUT / meta["preview"], optimize=True)
        metas.append(meta)
    (OUT / "avatars.json").write_text(json.dumps(metas, indent=2))
    print("wrote", len(metas), "avatars")


if __name__ == "__main__":
    main()
