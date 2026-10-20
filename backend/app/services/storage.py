import shutil
import uuid
from pathlib import Path

from ..config import get_settings


def root() -> Path:
    return get_settings().storage_dir.resolve()


def subdir(name: str) -> Path:
    p = root() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def url_for(rel: str | None) -> str | None:
    return f"/storage/{rel}" if rel else None


def new_id() -> str:
    return uuid.uuid4().hex


def remove_rel(rel: str | None) -> None:
    """Delete a file under storage given its relative path (path traversal safe)."""
    if not rel:
        return
    p = (root() / rel).resolve()
    if root() in p.parents and p.is_file():
        p.unlink(missing_ok=True)


def rmtree(p: Path) -> None:
    shutil.rmtree(p, ignore_errors=True)
