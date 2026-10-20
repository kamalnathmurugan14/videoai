from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..deps import get_current_user
from ..models import MediaAsset, Project, Scene, User
from ..services import media as media_svc, storage

router = APIRouter(prefix="/media", tags=["media"])


class AssetOut(BaseModel):
    id: str
    kind: str
    url: str
    width: int
    height: int
    name: str
    gender: str | None = None
    face_found: bool | None = None  # avatars: was a face detected (animation quality depends on it)
    background_removed: bool | None = None


def _out(a: MediaAsset) -> AssetOut:
    m = a.meta or {}
    return AssetOut(id=a.id, kind=a.kind, url=f"/storage/uploads/{a.filename}", width=a.width, height=a.height,
                    name=a.original_name, gender=m.get("gender"), face_found=m.get("found"), background_removed=m.get("cutout"))


@router.post("", response_model=AssetOut, status_code=201)
async def upload(kind: str = Form("image"), remove_bg: bool = Form(True), gender: str = Form("female"),
                 file: UploadFile = File(...), db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if kind not in ("image", "avatar"):
        raise HTTPException(422, "kind must be 'image' or 'avatar'")
    if gender not in ("female", "male"):
        raise HTTPException(422, "gender must be 'female' or 'male'")
    data = await file.read(get_settings().max_upload_mb * 1024 * 1024 + 1)
    meta = None
    if kind == "avatar":
        filename, w, h, meta = media_svc.save_avatar(data, remove_bg, gender)
    else:
        filename, w, h = media_svc.save_image(data)
    a = MediaAsset(id=filename.split(".")[0], owner_id=user.id, kind=kind, filename=filename,
                   original_name=(file.filename or "")[:255], width=w, height=h, meta=meta)
    db.add(a)
    db.commit()
    return _out(a)


@router.get("", response_model=list[AssetOut])
def list_assets(kind: str | None = None, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    q = select(MediaAsset).where(MediaAsset.owner_id == user.id).order_by(MediaAsset.created_at.desc())
    if kind:
        q = q.where(MediaAsset.kind == kind)
    return [_out(a) for a in db.scalars(q)]


@router.delete("/{asset_id}", status_code=204)
def delete_asset(asset_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    a = db.get(MediaAsset, asset_id)
    if not a or a.owner_id != user.id:
        raise HTTPException(404, "Not found")
    for sc in db.scalars(select(Scene).where(Scene.media_asset_id == asset_id)):
        sc.media_asset_id = None
    for p in db.scalars(select(Project).where(Project.owner_id == user.id)):
        if p.avatar_id == f"asset:{asset_id}":
            p.avatar_id = "aria"
    storage.remove_rel(f"uploads/{a.filename}")
    db.delete(a)
    db.commit()
