from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..deps import client_ip, get_current_user
from ..models import MediaAsset, Project, RenderJob, Scene, User
from ..schemas.auth import UsageOut
from ..schemas.project import JobOut, ProjectCreate, ProjectOut, ProjectSummary, ProjectUpdate
from ..services import catalog, projects as svc, remote_gpu, sadtalker, storage

router = APIRouter(tags=["projects"])
ACTIVE = ("queued", "rendering")


def get_owned(db: Session, user: User, project_id: int) -> Project:
    p = db.get(Project, project_id)
    if not p or p.owner_id != user.id:
        raise HTTPException(404, "Project not found")
    return p


def _validate_assets(db: Session, user: User, asset_ids: set[str]) -> None:
    if not asset_ids:
        return
    found = set(db.scalars(select(MediaAsset.id).where(MediaAsset.id.in_(asset_ids), MediaAsset.owner_id == user.id)))
    if found != asset_ids:
        raise HTTPException(422, "Unknown or foreign media asset")


@router.get("/projects", response_model=list[ProjectSummary])
def list_projects(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    rows = db.scalars(select(Project).where(Project.owner_id == user.id).order_by(Project.updated_at.desc())).all()
    return [svc.to_summary(p) for p in rows]


@router.post("/projects", response_model=ProjectOut, status_code=201)
def create_project(body: ProjectCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    p = Project(owner_id=user.id, title=body.title.strip() or "Untitled video", background=svc.DEFAULT_BG)
    try:
        svc.apply_template(p, body.template_id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    db.add(p)
    db.commit()
    return svc.to_out(p)


@router.get("/projects/{project_id}", response_model=ProjectOut)
def get_project(project_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return svc.to_out(get_owned(db, user, project_id))


@router.put("/projects/{project_id}", response_model=ProjectOut)
def update_project(project_id: int, body: ProjectUpdate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    p = get_owned(db, user, project_id)
    s = get_settings()
    data = body.model_dump(exclude_unset=True)
    asset_ids: set[str] = set()
    if body.avatar_id and body.avatar_id.startswith("asset:"):
        asset_ids.add(body.avatar_id[6:])
    elif body.avatar_id and not catalog.get_avatar(body.avatar_id):
        raise HTTPException(422, "Unknown avatar")
    if body.voice and body.voice not in catalog.VOICE_IDS:
        raise HTTPException(422, "Unknown voice")
    if body.music_id:
        from ..services import music
        if body.music_id not in music.TRACKS:
            raise HTTPException(422, "Unknown music track")
    bgs = [body.background] if body.background else []
    if body.scenes is not None:
        bgs += [sc.background for sc in body.scenes if sc.background]
    for bg in bgs:
        if bg.type == "scenery" and not catalog.get_scenery(bg.scenery_id or ""):
            raise HTTPException(422, "Unknown scenery")
    if body.background and body.background.asset_id:
        asset_ids.add(body.background.asset_id)
    if body.scenes is not None:
        if len(body.scenes) > s.max_scenes:
            raise HTTPException(422, f"At most {s.max_scenes} scenes")
        if any(len(sc.text) > s.max_scene_chars for sc in body.scenes):
            raise HTTPException(422, f"Each scene is limited to {s.max_scene_chars} characters")
        if sum(len(sc.text) for sc in body.scenes) > s.max_total_chars:
            raise HTTPException(422, f"Script is limited to {s.max_total_chars} characters in total")
        for sc in body.scenes:
            if sc.media_asset_id:
                asset_ids.add(sc.media_asset_id)
            if sc.background and sc.background.asset_id:
                asset_ids.add(sc.background.asset_id)
    _validate_assets(db, user, asset_ids)

    for field in ("title", "aspect_ratio", "avatar_id", "voice", "speed", "captions_enabled", "caption_style", "music_volume"):
        if data.get(field) is not None:
            setattr(p, field, data[field])
    if body.clear_music:
        p.music_id = None
    elif body.music_id:
        p.music_id = body.music_id
    if body.background is not None:
        p.background = body.background.model_dump()
    if body.intro_enabled is not None:
        p.intro_enabled = body.intro_enabled
    if body.gestures_enabled is not None:
        p.gestures_enabled = body.gestures_enabled
    if body.animation_engine is not None:
        if body.animation_engine == "sadtalker":
            st = sadtalker.status()
            if not st["ready"]:
                raise HTTPException(422, "SadTalker is not installed on this server: " + "; ".join(st["problems"]))
        if body.animation_engine.startswith("gpu_") and not remote_gpu.configured():
            raise HTTPException(422, "No GPU worker connected: start the Colab notebook and paste its URL on the System check page")
        p.animation_engine = body.animation_engine
    if body.framing is not None:
        p.framing = body.framing
    if body.intro_title is not None:
        p.intro_title = body.intro_title.strip() or None
    if body.scenes is not None:
        p.scenes = [
            Scene(position=i, text=sc.text, media_asset_id=sc.media_asset_id,
                  background=sc.background.model_dump() if sc.background else None)
            for i, sc in enumerate(body.scenes)
        ]
    if p.status in ("done", "failed"):
        pass  # keep last render available; user can re-render
    db.commit()
    return svc.to_out(p)


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    p = get_owned(db, user, project_id)
    if p.status in ACTIVE:
        raise HTTPException(409, "Cannot delete while rendering")
    storage.remove_rel(p.video_path)
    storage.remove_rel(p.thumbnail_path)
    db.delete(p)
    db.commit()


@router.post("/projects/{project_id}/duplicate", response_model=ProjectOut, status_code=201)
def duplicate_project(project_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    src = get_owned(db, user, project_id)
    p = Project(
        owner_id=user.id, title=f"{src.title} (copy)"[:120], aspect_ratio=src.aspect_ratio, avatar_id=src.avatar_id,
        voice=src.voice, speed=src.speed, captions_enabled=src.captions_enabled, caption_style=src.caption_style,
        music_id=src.music_id, music_volume=src.music_volume, background=src.background,
        intro_enabled=src.intro_enabled, intro_title=src.intro_title, gestures_enabled=src.gestures_enabled, animation_engine=src.animation_engine, framing=src.framing,
        scenes=[Scene(position=s.position, text=s.text, background=s.background, media_asset_id=s.media_asset_id) for s in src.scenes],
    )
    db.add(p)
    db.commit()
    return svc.to_out(p)


@router.post("/projects/{project_id}/render", response_model=JobOut, status_code=202)
def start_render(project_id: int, request: Request, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    p = get_owned(db, user, project_id)
    if p.status in ACTIVE:
        raise HTTPException(409, "This video is already rendering")
    if not any(s.text.strip() for s in p.scenes):
        raise HTTPException(422, "Add some script text to at least one scene first")
    ip = client_ip(request)
    try:
        svc.check_quota(db, user.id, ip)
    except PermissionError as e:
        raise HTTPException(429, str(e))
    job = RenderJob(project_id=p.id, user_id=user.id, ip=ip)
    p.status, p.error = "queued", None
    db.add(job)
    db.commit()
    return job


@router.get("/projects/{project_id}/job", response_model=JobOut | None)
def latest_job(project_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    p = get_owned(db, user, project_id)
    return p.jobs[-1] if p.jobs else None


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    job = db.get(RenderJob, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/projects/{project_id}/download")
def download(project_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    p = get_owned(db, user, project_id)
    path = (storage.root() / p.video_path) if p.video_path else None
    if not path or not path.is_file():
        raise HTTPException(404, "No rendered video yet")
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in p.title).strip() or "video"
    return FileResponse(path, media_type="video/mp4", filename=f"{safe}.mp4")


@router.get("/usage", response_model=UsageOut)
def usage(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return UsageOut(used_today=svc.renders_today(db, user.id), limit=get_settings().render_daily_limit_user)
