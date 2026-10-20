"""Generates the stylised, softly-blurred scenery backgrounds in assets/sceneries (run from backend/)."""
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "assets" / "sceneries"
W, H = 960, 540


def vgrad(top, bottom, h=H, w=W):
    im = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(im)
    for y in range(h):
        t = y / (h - 1)
        d.line([(0, y), (w, y)], fill=tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)))
    return im


def finish(im, blur=1.6):
    return im.resize((1920, 1080), Image.LANCZOS).filter(ImageFilter.GaussianBlur(blur))


def office():
    im = vgrad((236, 228, 214), (205, 190, 168))
    d = ImageDraw.Draw(im)
    # big window with city skyline
    d.rectangle([150, 60, 810, 360], fill=(150, 200, 235))
    sky = vgrad((140, 195, 240), (225, 240, 250), 300, 660)
    im.paste(sky, (150, 60))
    rnd = random.Random(3)
    x = 150
    while x < 810:
        w = rnd.randint(40, 90)
        h = rnd.randint(60, 200)
        d.rectangle([x, 360 - h, min(x + w, 810), 360], fill=(110 + rnd.randint(0, 40), 130 + rnd.randint(0, 30), 160 + rnd.randint(0, 30)))
        x += w + 4
    for gx in (370, 590):
        d.rectangle([gx - 4, 60, gx + 4, 360], fill=(240, 236, 228))
    d.rectangle([140, 52, 820, 66], fill=(240, 236, 228))
    d.rectangle([140, 354, 820, 372], fill=(240, 236, 228))
    # shelf + plant + desk
    d.rectangle([20, 150, 120, 160], fill=(150, 110, 80))
    for i, c in enumerate([(200, 80, 80), (80, 120, 200), (230, 190, 80), (90, 160, 110)]):
        d.rectangle([28 + i * 22, 100, 44 + i * 22, 150], fill=c)
    d.polygon([(850, 330), (870, 440), (930, 440), (950, 330)], fill=(170, 100, 70))
    for a in range(-60, 70, 20):
        d.ellipse([900 + a * 0.9 - 28, 250 - abs(a) * 0.8 - 30, 900 + a * 0.9 + 28, 250 - abs(a) * 0.8 + 30], fill=(70, 140, 80))
    d.rectangle([0, 450, W, H], fill=(140, 105, 78))
    d.rectangle([0, 440, W, 456], fill=(165, 128, 96))
    return finish(im)


def beach():
    im = vgrad((255, 140, 90), (255, 214, 150), 300)
    full = Image.new("RGB", (W, H))
    full.paste(im, (0, 0))
    d = ImageDraw.Draw(full)
    d.ellipse([650, 170, 770, 290], fill=(255, 240, 190))
    for y in range(300, 400):  # sea
        t = (y - 300) / 100
        d.line([(0, y), (W, y)], fill=(int(70 + 40 * t), int(140 + 20 * t), int(190 - 20 * t)))
    for y in range(300, 400, 14):
        d.line([(0, y), (W, y)], fill=(255, 220, 170), width=1)
    d.rectangle([0, 400, W, H], fill=(238, 205, 150))
    d.polygon([(0, 400), (W, 380), (W, 410), (0, 430)], fill=(250, 225, 175))
    d.line([(120, 540), (160, 300)], fill=(70, 45, 30), width=10)
    for ang in (-150, -110, -70, -30, 10):
        ex, ey = 160 + 110 * math.cos(math.radians(ang)), 300 + 80 * math.sin(math.radians(ang)) * -1
        d.line([(160, 300), (ex, ey - 20)], fill=(40, 90, 50), width=8)
    return finish(full, 2.0)


def city_night():
    im = vgrad((25, 20, 70), (120, 60, 130))
    d = ImageDraw.Draw(im)
    d.ellipse([740, 60, 800, 120], fill=(250, 245, 220))
    rnd = random.Random(9)
    for layer, (col, hmax) in enumerate([((50, 40, 100), 300), ((30, 25, 70), 240)]):
        x = -20
        while x < W:
            w, h = rnd.randint(50, 110), rnd.randint(80, hmax)
            top = H - h - layer * 0
            d.rectangle([x, top, x + w, H], fill=col)
            for wy in range(top + 10, H - 10, 22):
                for wx in range(x + 8, x + w - 8, 16):
                    if rnd.random() < 0.35:
                        d.rectangle([wx, wy, wx + 7, wy + 11], fill=(255, 214, 120))
            x += w + 2
    return finish(im, 2.2)


def forest():
    im = vgrad((190, 225, 210), (240, 245, 225))
    d = ImageDraw.Draw(im)
    rnd = random.Random(5)
    for layer, col in enumerate([(120, 170, 140), (80, 140, 105), (40, 105, 70)]):
        base = 300 + layer * 70
        d.polygon([(0, H)] + [(x, base - 40 * math.sin(x / (90 + layer * 30)) - rnd.randint(0, 6)) for x in range(0, W + 20, 20)] + [(W, H)], fill=col)
        for i in range(10 + layer * 6):
            tx = rnd.randint(0, W)
            th = 80 + layer * 50 + rnd.randint(0, 40)
            tb = base + 20 + layer * 20
            for k in range(3):
                d.polygon([(tx - 30 - layer * 10 + k * 6, tb - k * th * 0.28), (tx + 30 + layer * 10 - k * 6, tb - k * th * 0.28), (tx, tb - th * 0.5 - k * th * 0.28)], fill=tuple(max(0, c - 10 - k * 4) for c in col))
    rays = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rays)
    for x in range(100, 900, 160):
        rd.polygon([(x, 0), (x + 60, 0), (x + 260, H), (x + 120, H)], fill=(255, 255, 220, 38))
    im = Image.alpha_composite(im.convert("RGBA"), rays).convert("RGB")
    return finish(im, 2.0)


def mountains():
    im = vgrad((120, 170, 230), (240, 225, 215))
    d = ImageDraw.Draw(im)
    rnd = random.Random(2)
    for layer, (col, base) in enumerate([((150, 160, 195), 330), ((105, 118, 160), 380), ((70, 85, 125), 430)]):
        pts = [(0, H)]
        x = 0
        while x <= W + 60:
            pts.append((x, base - rnd.randint(40, 170 - layer * 25)))
            x += rnd.randint(70, 140)
        pts.append((W, H))
        d.polygon(pts, fill=col)
        if layer == 0:
            for (px, py) in pts[1:-1]:
                d.polygon([(px - 26, py + 40), (px, py), (px + 26, py + 40)], fill=(245, 248, 255))
    d.rectangle([0, 450, W, H], fill=(70, 125, 160))
    for y in range(452, H, 12):
        d.line([(0, y), (W, y)], fill=(110, 160, 190))
    return finish(im, 2.2)


def studio():
    im = vgrad((15, 40, 62), (25, 80, 96))
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for x, a in ((200, 55), (480, 70), (760, 55)):
        d.polygon([(x - 18, 0), (x + 18, 0), (x + 190, H), (x - 190, H)], fill=(180, 230, 255, a))
    for i in range(14):
        cx = 60 + i * 66
        d.ellipse([cx - 14, 410 + (i % 3) * 18, cx + 14, 438 + (i % 3) * 18], fill=(120, 220, 230, 90))
    d.ellipse([-100, 440, W + 100, H + 200], fill=(10, 30, 45, 160))
    return finish(Image.alpha_composite(im.convert("RGBA"), over).convert("RGB"), 2.4)


def library():
    im = vgrad((110, 70, 45), (70, 42, 28))
    d = ImageDraw.Draw(im)
    rnd = random.Random(4)
    for row in range(5):
        y0 = 30 + row * 100
        d.rectangle([0, y0 + 88, W, y0 + 100], fill=(60, 36, 22))
        x = 8
        while x < W - 20:
            bw, bh = rnd.randint(18, 34), rnd.randint(55, 84)
            col = rnd.choice([(170, 60, 50), (60, 100, 150), (210, 170, 70), (70, 120, 90), (130, 90, 150), (220, 200, 170)])
            d.rectangle([x, y0 + 88 - bh, x + bw, y0 + 88], fill=col)
            d.line([(x + 4, y0 + 88 - bh + 8), (x + bw - 4, y0 + 88 - bh + 8)], fill=(250, 240, 200), width=2)
            x += bw + rnd.randint(1, 4)
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([260, 60, 700, 520], fill=(255, 200, 120, 60))
    im = Image.alpha_composite(im.convert("RGBA"), glow.filter(ImageFilter.GaussianBlur(40))).convert("RGB")
    return finish(im, 2.6)


def tech():
    im = vgrad((8, 14, 40), (30, 20, 90))
    d = ImageDraw.Draw(im)
    for x in range(0, W, 48):
        d.line([(x, 0), (x, H)], fill=(30, 50, 110))
    for y in range(0, H, 48):
        d.line([(0, y), (W, y)], fill=(30, 50, 110))
    rnd = random.Random(8)
    nodes = [(rnd.randint(40, W - 40), rnd.randint(40, H - 40)) for _ in range(26)]
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if math.dist(a, b) < 190:
                d.line([a, b], fill=(70, 140, 230), width=1)
    for (x, y) in nodes:
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(110, 220, 255))
    return finish(im, 1.8)


SCENES = [
    ("office", "Modern office", office), ("studio", "Studio lights", studio), ("city", "City at night", city_night),
    ("beach", "Beach sunset", beach), ("forest", "Forest", forest), ("mountains", "Mountains & lake", mountains),
    ("library", "Library", library), ("tech", "Tech network", tech),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = []
    for sid, name, fn in SCENES:
        fn().save(OUT / f"{sid}.jpg", "JPEG", quality=82, optimize=True)
        meta.append({"id": sid, "name": name, "file": f"{sid}.jpg"})
    (OUT / "sceneries.json").write_text(json.dumps(meta, indent=2))
    print("wrote", len(meta))


if __name__ == "__main__":
    main()
