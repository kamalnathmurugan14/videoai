"""Caption timing and rendering (Pillow)."""
import glob
import re
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

from ..config import get_settings

_FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", "/System/Library/Fonts", "C:/Windows/Fonts"]
_SCRIPT_FONTS = [  # (regex of chars, filename globs)
    (r"[\u0900-\u097F]", ["NotoSansDevanagari-Bold*", "NotoSansDevanagari*"]),
    (r"[\u0B80-\u0BFF]", ["NotoSansTamil-Bold*", "NotoSansTamil*"]),
    (r"[\u0600-\u06FF]", ["NotoSansArabic-Bold*", "NotoSansArabic*", "NotoNaskhArabic*"]),
    (r"[\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]", ["NotoSansCJK*Bold*", "NotoSansCJK*", "wqy*", "*gothic*"]),
]
_LATIN = ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "NotoSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf", "Inter*Bold*"]


def _find(patterns: list[str]) -> str | None:
    for pat in patterns:
        for d in _FONT_DIRS:
            hits = glob.glob(f"{d}/**/{pat}", recursive=True)
            if hits:
                return sorted(hits)[0]
    return None


@lru_cache(maxsize=64)
def _font_path(script_key: int) -> str | None:
    if get_settings().caption_font:
        return get_settings().caption_font
    if script_key >= 0:
        p = _find(_SCRIPT_FONTS[script_key][1])
        if p:
            return p
    return _find(_LATIN)


@lru_cache(maxsize=128)
def get_font(text_sample: str, size: int) -> ImageFont.FreeTypeFont:
    key = next((i for i, (rx, _) in enumerate(_SCRIPT_FONTS) if re.search(rx, text_sample)), -1)
    path = _font_path(key)
    try:
        return ImageFont.truetype(path, size) if path else ImageFont.load_default(size)
    except OSError:
        return ImageFont.load_default(size)


def split_words(text: str) -> tuple[list[str], str]:
    """Return (tokens, joiner). Languages without spaces are cut into short groups."""
    text = " ".join(text.split())
    if " " not in text and len(text) > 14:
        return [text[i:i + 8] for i in range(0, len(text), 8)], ""
    return text.split(" "), " "


def word_timings(words: list[str], start: float, end: float) -> list[tuple[float, float]]:
    """Distribute speech time across words by length, with extra weight after punctuation."""
    weights = [len(w) + 2 + (4 if re.search(r"[.,!?;:。！？]$", w) else 0) for w in words]
    total = sum(weights) or 1
    t, out = start, []
    for w in weights:
        d = (end - start) * w / total
        out.append((t, t + d))
        t += d
    return out


def chunk_words(words: list[str], max_words: int = 5, max_chars: int = 30) -> list[list[int]]:
    chunks, cur, n = [], [], 0
    for i, w in enumerate(words):
        if cur and (len(cur) >= max_words or n + len(w) + 1 > max_chars):
            chunks.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += len(w) + 1
        if re.search(r"[.!?。！？]$", w):
            chunks.append(cur)
            cur, n = [], 0
    if cur:
        chunks.append(cur)
    return chunks


class CaptionRenderer:
    def __init__(self, width: int, height: int, style: str, text: str):
        self.w, self.h, self.style = width, height, style
        ratio = 0.04 if width > height else 0.058
        self.size = max(14, int(width * ratio * (0.9 if style == "minimal" else 1.0)))
        self.font = get_font(text, self.size)

    def render(self, words: list[str], joiner: str, active: int = -1) -> Image.Image:
        """RGBA image covering the full frame width (height = caption block only)."""
        d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        space = d0.textlength(joiner or " ", font=self.font) if joiner else 0
        max_w = self.w * 0.86
        lines, cur, cw = [], [], 0.0
        for i, w in enumerate(words):
            ww = d0.textlength(w, font=self.font)
            if cur and cw + space + ww > max_w:
                lines.append(cur)
                cur, cw = [], 0.0
            cur.append((i, w, ww))
            cw += ww + (space if cur[:-1] else 0)
        lines.append(cur)
        lh = int(self.size * 1.3)
        pad = int(self.size * 0.5)
        img = Image.new("RGBA", (self.w, lh * len(lines) + pad * 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        widths = [sum(x[2] for x in ln) + space * (len(ln) - 1) for ln in lines]
        if self.style == "classic":
            bw = max(widths) + pad * 2
            d.rounded_rectangle([(self.w - bw) / 2, 0, (self.w + bw) / 2, img.height], radius=int(self.size * 0.4), fill=(15, 15, 25, 200))
        for li, ln in enumerate(lines):
            x = (self.w - widths[li]) / 2
            y = pad + li * lh
            for i, w, ww in ln:
                color = (255, 255, 255, 255)
                if self.style == "bold":
                    if i == active:
                        color = (255, 214, 10, 255)
                    d.text((x, y), w, font=self.font, fill=color, stroke_width=max(2, self.size // 9), stroke_fill=(0, 0, 0, 255))
                elif self.style == "minimal":
                    d.text((x + 2, y + 2), w, font=self.font, fill=(0, 0, 0, 150))
                    d.text((x, y), w, font=self.font, fill=color)
                else:
                    d.text((x, y), w, font=self.font, fill=color)
                x += ww + space
        return img
