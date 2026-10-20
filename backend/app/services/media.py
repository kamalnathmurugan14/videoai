"""Image upload handling: validate with Pillow and re-encode (strips metadata / polyglot payloads)."""
import io

from fastapi import HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import get_settings
from . import face, storage

Image.MAX_IMAGE_PIXELS = 40_000_000


def _open(data: bytes) -> Image.Image:
    s = get_settings()
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {s.max_upload_mb} MB)")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "Not a valid image (use PNG, JPEG or WebP)")
    if img.format not in ("PNG", "JPEG", "WEBP"):
        raise HTTPException(400, "Unsupported image type (use PNG, JPEG or WebP)")
    return ImageOps.exif_transpose(img)


def save_image(data: bytes) -> tuple[str, int, int]:
    """Scene/background image. Returns (filename, width, height)."""
    img = _open(data)
    img.thumbnail((1600, 1600))
    name = storage.new_id()
    if img.mode in ("RGBA", "LA", "P"):
        filename = f"{name}.png"
        img.convert("RGBA").save(storage.subdir("uploads") / filename, "PNG")
    else:
        filename = f"{name}.jpg"
        img.convert("RGB").save(storage.subdir("uploads") / filename, "JPEG", quality=88)
    return filename, img.width, img.height


def save_avatar(data: bytes, remove_bg: bool, gender: str) -> tuple[str, int, int, dict]:
    """Photo -> normalised avatar canvas with detected landmarks. Returns (filename, w, h, meta)."""
    img = _open(data)
    canvas, meta, full = face.build_avatar_full(img, remove_bg, gender)
    name = storage.new_id()
    filename = f"{name}.png"
    canvas.save(storage.subdir("uploads") / filename, "PNG", optimize=True)
    full.save(storage.subdir("uploads") / f"{name}_full.jpg", "JPEG", quality=92)  # original-photo framing
    meta["full"]["file"] = f"{name}_full.jpg"
    return filename, canvas.width, canvas.height, meta
