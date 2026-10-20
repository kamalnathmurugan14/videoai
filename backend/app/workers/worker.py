"""DB-backed job queue worker. Runs in-process (thread) by default or standalone: python -m app.workers.run"""
import logging
import shutil
import threading
import traceback
from pathlib import Path

from sqlalchemy import select, update

from ..config import ASSETS_DIR, get_settings
from ..db import SessionLocal
from ..models import MediaAsset, Project, RenderJob
from ..models.base import utcnow
from ..services import catalog, music, renderer, storage

log = logging.getLogger("videoai.worker")


def recover_stale() -> None:
    """Jobs left 'running' by a crashed process go back in the queue."""
    with SessionLocal() as db:
        stuck = db.scalars(select(RenderJob).where(RenderJob.status == "running")).all()
        for j in stuck:
            j.status, j.stage = "queued", "Re-queued after restart"
            if j.project:
                j.project.status = "queued"
        db.commit()


def claim_next() -> int | None:
    with SessionLocal() as db:
        job = db.scalar(select(RenderJob).where(RenderJob.status == "queued").order_by(RenderJob.id).limit(1))
        if not job:
            return None
        res = db.execute(update(RenderJob).where(RenderJob.id == job.id, RenderJob.status == "queued")
                         .values(status="running", started_at=utcnow(), stage="Starting", progress=1))
        if res.rowcount != 1:  # another worker got it
            db.rollback()
            return None
        db.get(Project, job.project_id).status = "rendering"
        db.commit()
        return job.id


def build_spec(db, project: Project) -> tuple[renderer.RenderSpec, dict[str, Path]]:
    assets = {a.id: a for a in db.scalars(select(MediaAsset).where(MediaAsset.owner_id == project.owner_id))}
    up = storage.subdir("uploads")
    paths = {aid: up / a.filename for aid, a in assets.items()}
    avatar_meta: dict = {}
    if project.avatar_id.startswith("asset:") and project.avatar_id[6:] in paths:
        avatar_path = paths[project.avatar_id[6:]]
    else:
        av = catalog.get_avatar(project.avatar_id) or catalog.load_avatars()[0]
        avatar_path, avatar_meta = ASSETS_DIR / "avatars" / av["image"], av
    default_bg = project.background or {}
    scenes = [
        renderer.SceneSpec(text=s.text, background=s.background or default_bg, media_path=paths.get(s.media_asset_id or ""))
        for s in project.scenes
    ]
    music_path = None
    if project.music_id:
        music.ensure_library()
        music_path = music.path_for(project.music_id)
    spec = renderer.RenderSpec(
        scenes=scenes, aspect_ratio=project.aspect_ratio, avatar_path=avatar_path, avatar_meta=avatar_meta,
        voice=project.voice, speed=project.speed, captions_enabled=project.captions_enabled,
        caption_style=project.caption_style, music_path=music_path, music_volume=project.music_volume,
    )
    return spec, paths


def process_job(job_id: int) -> None:
    work = storage.subdir("tmp") / f"job-{job_id}"
    with SessionLocal() as db:
        job = db.get(RenderJob, job_id)
        project = db.get(Project, job.project_id)
        spec, bg_paths = build_spec(db, project)
        pid, old_video, old_thumb = project.id, project.video_path, project.thumbnail_path

    def progress(pct: int, stage: str) -> None:
        with SessionLocal() as s:
            s.execute(update(RenderJob).where(RenderJob.id == job_id).values(progress=pct, stage=stage))
            s.commit()

    try:
        rel_video, rel_thumb = f"renders/p{pid}-j{job_id}.mp4", f"thumbs/p{pid}-j{job_id}.jpg"
        storage.subdir("renders"), storage.subdir("thumbs")
        res = renderer.render_video(spec, work, storage.root() / rel_video, storage.root() / rel_thumb, progress, bg_paths)
        with SessionLocal() as db:
            job, project = db.get(RenderJob, job_id), db.get(Project, pid)
            job.status, job.progress, job.stage, job.finished_at = "done", 100, "Done", utcnow()
            job.info = "\n".join(res.notes)
            if project:
                project.status, project.error = "done", None
                project.video_path, project.thumbnail_path, project.duration = rel_video, rel_thumb, res.duration
            db.commit()
        storage.remove_rel(old_video)
        storage.remove_rel(old_thumb)
    except Exception as e:
        log.error("render job %s failed: %s", job_id, traceback.format_exc())
        msg = str(e)[:500] or e.__class__.__name__
        with SessionLocal() as db:
            job, project = db.get(RenderJob, job_id), db.get(Project, pid)
            job.status, job.stage, job.error, job.finished_at = "failed", "Failed", msg, utcnow()
            if project:
                project.status, project.error = "failed", msg
            db.commit()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def process_next() -> bool:
    job_id = claim_next()
    if job_id is None:
        return False
    process_job(job_id)
    return True


class WorkerThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="render-worker")
        self.stop_event = threading.Event()

    def run(self):
        recover_stale()
        while not self.stop_event.is_set():
            try:
                if process_next():
                    continue
            except Exception:
                log.exception("worker loop error")
            self.stop_event.wait(get_settings().worker_poll_seconds)
