"""Project helpers shared by API and worker: serialization, template application, render quota."""
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import Project, RenderJob, Scene
from ..schemas.project import Background, JobOut, ProjectOut, ProjectSummary, SceneOut
from . import catalog, storage

DEFAULT_BG = {"type": "gradient", "color": "#4f46e5", "color2": "#9333ea", "asset_id": None}


def summary_fields(p: Project) -> dict:
    return dict(
        id=p.id, title=p.title, status=p.status, aspect_ratio=p.aspect_ratio, duration=p.duration,
        video_url=storage.url_for(p.video_path), thumbnail_url=storage.url_for(p.thumbnail_path),
        scene_count=len(p.scenes), created_at=p.created_at, updated_at=p.updated_at,
    )


def to_summary(p: Project) -> ProjectSummary:
    return ProjectSummary(**summary_fields(p))


def to_out(p: Project) -> ProjectOut:
    job = p.jobs[-1] if p.jobs else None
    return ProjectOut(
        **summary_fields(p),
        avatar_id=p.avatar_id, voice=p.voice, speed=p.speed, captions_enabled=p.captions_enabled,
        caption_style=p.caption_style, music_id=p.music_id, music_volume=p.music_volume,
        background=Background(**(p.background or DEFAULT_BG)), error=p.error,
        scenes=[SceneOut.model_validate(s) for s in p.scenes],
        latest_job=JobOut.model_validate(job) if job else None,
    )


def apply_template(p: Project, template_id: str | None) -> None:
    tpl = next((t for t in catalog.TEMPLATES if t["id"] == template_id), None)
    if template_id and not tpl:
        raise ValueError("Unknown template")
    if tpl:
        p.aspect_ratio = tpl["aspect_ratio"]
        p.background = tpl["background"]
        p.scenes = [Scene(position=i, text=t) for i, t in enumerate(tpl["scenes"])]
    else:
        p.scenes = [Scene(position=0, text="")]


def start_of_utc_day() -> datetime:
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)


def renders_today(db: Session, user_id: int) -> int:
    return db.scalar(select(func.count(RenderJob.id)).where(RenderJob.user_id == user_id, RenderJob.created_at >= start_of_utc_day())) or 0


def check_quota(db: Session, user_id: int, ip: str) -> None:
    s = get_settings()
    if renders_today(db, user_id) >= s.render_daily_limit_user:
        raise PermissionError(f"Daily render limit reached ({s.render_daily_limit_user} per day). Try again tomorrow.")
    by_ip = db.scalar(select(func.count(RenderJob.id)).where(RenderJob.ip == ip, RenderJob.created_at >= start_of_utc_day())) or 0
    if by_ip >= s.render_daily_limit_ip:
        raise PermissionError("Daily render limit reached for your network. Try again tomorrow.")
