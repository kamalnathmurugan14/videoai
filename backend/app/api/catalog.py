import hashlib
import json
import time
from collections import defaultdict, deque

from fastapi import APIRouter, Depends, HTTPException, Request

from ..config import get_settings
from ..deps import client_ip, get_current_user
from ..models import User
from ..schemas.misc import ImportText, ImportUrl, VoicePreview
from ..services import catalog, landmarks, music, remote_gpu, sadtalker, script_import, storage, tts

router = APIRouter(tags=["catalog"])
_hits: dict[str, deque] = defaultdict(deque)


def _rate_limit(key: str, limit: int, window: int = 60) -> None:
    q, now = _hits[key], time.time()
    while q and q[0] < now - window:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "Too many requests, slow down a little")
    q.append(now)


@router.get("/config")
def config():
    s = get_settings()
    return {
        "aspects": catalog.ASPECTS, "caption_styles": catalog.CAPTION_STYLES, "gradients": catalog.GRADIENTS,
        "limits": {"max_scenes": s.max_scenes, "max_scene_chars": s.max_scene_chars, "max_total_chars": s.max_total_chars,
                   "max_upload_mb": s.max_upload_mb, "daily_renders": s.render_daily_limit_user},
        "llm_enabled": bool(s.anthropic_api_key),
        "face_tracking": landmarks._interpreters() is not None,
        "sadtalker_ready": sadtalker.status()["ready"],
        "remote_gpu": remote_gpu.configured(),
    }


@router.get("/avatars")
def avatars():
    return [{"id": a["id"], "name": a["name"], "gender": a.get("gender"), "url": f"/avatar-art/{a['preview']}"}
            for a in catalog.load_avatars()]


@router.get("/sceneries")
def sceneries():
    return [{"id": s["id"], "name": s["name"], "url": f"/scenery-art/{s['file']}"} for s in catalog.load_sceneries()]


@router.get("/voices")
def voices():
    return [{"id": v, "name": n, "language": lang, "gender": g} for v, n, lang, g in catalog.VOICES]


@router.get("/templates")
def templates():
    return catalog.TEMPLATES


@router.get("/music")
def music_list():
    return music.listing()


@router.post("/voices/preview")
def voice_preview(body: VoicePreview, request: Request, user: User = Depends(get_current_user)):
    if body.voice not in catalog.VOICE_IDS:
        raise HTTPException(422, "Unknown voice")
    _rate_limit(f"preview:{user.id}", 15)
    key = hashlib.sha1(f"{get_settings().tts_providers}|{body.voice}|{body.speed}|{body.text}".encode()).hexdigest()[:20]
    out = storage.subdir("previews") / f"{key}.mp3"
    info = out.with_suffix(".txt")
    if not out.exists():
        wav = out.with_suffix(".wav")
        try:
            engine, errors = tts.synthesize(body.text, body.voice, body.speed, wav)
            tts._ffmpeg(["-i", str(wav), "-b:a", "64k", str(out)])
            warning = None
            if errors:
                warning = f"The preferred voice engine failed ({errors[0]}). Using '{engine}', so the voice may differ from the one you picked."
            elif engine == "gtts" and tts.voice_gender(body.voice) == "male":
                warning = "Google TTS only has a female voice, so this male voice cannot be played."
            info.write_text(json.dumps({"engine": engine, "warning": warning}))
        except tts.TTSError as e:
            raise HTTPException(503, str(e))
        finally:
            wav.unlink(missing_ok=True)
    meta = json.loads(info.read_text()) if info.exists() else {}
    return {"url": f"/storage/previews/{key}.mp3", **meta}


@router.post("/import/text")
def import_text(body: ImportText, user: User = Depends(get_current_user)):
    _rate_limit(f"import:{user.id}", 10)
    scenes, method = script_import.to_scenes(body.text, body.max_scenes)
    return {"scenes": scenes, "method": method, "title": None}


@router.post("/import/url")
def import_url(body: ImportUrl, user: User = Depends(get_current_user)):
    _rate_limit(f"import:{user.id}", 10)
    try:
        title, text = script_import.fetch_url_text(str(body.url))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        raise HTTPException(400, "Could not fetch that URL")
    if len(text) < 40:
        raise HTTPException(422, "Couldn't find enough readable text on that page")
    scenes, method = script_import.to_scenes(text, body.max_scenes)
    return {"scenes": scenes, "method": method, "title": title[:120] or None}
