"""System check: lets a user confirm which build is running and which optional engines work on their machine."""
import shutil
import subprocess
import time
from pathlib import Path

from fastapi import APIRouter, Depends

from ..config import APP_VERSION, get_settings
from ..deps import get_current_user
from ..models import User
from ..services import catalog, landmarks, sadtalker, segment, storage, tts
from .catalog import _rate_limit

router = APIRouter(prefix="/system", tags=["system"])


def _git_commit() -> str | None:
    head = Path(__file__).resolve().parents[3] / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref:"):
            return (head.parent / ref[5:]).read_text().strip()[:7]
        return ref[:7]
    except OSError:
        return None


@router.get("")
def system_info():
    s = get_settings()
    ffmpeg = shutil.which(s.ffmpeg_bin)
    ffmpeg_version = None
    if ffmpeg:
        try:
            ffmpeg_version = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, timeout=10).stdout.split("\n")[0][:80]
        except Exception:
            pass
    return {
        "version": APP_VERSION,
        "commit": _git_commit(),
        "face_tracking": landmarks._interpreters() is not None,  # precise 478-point landmarks (else Haar fallback)
        "segmentation": segment.available(),  # person cut-out model (else GrabCut fallback)
        "ffmpeg": ffmpeg_version,
        "tts_order": s.tts_list,
        "espeak": bool(shutil.which("espeak-ng") or shutil.which("espeak")),
        "sadtalker": sadtalker.status(),
    }


@router.post("/voice-check")
def voice_check(user: User = Depends(get_current_user)):
    """Synthesise a short phrase with every configured engine on this machine and report what works (and why not)."""
    _rate_limit(f"voicecheck:{user.id}", 3)
    out = []
    tmp = storage.subdir("tmp")
    for name in get_settings().tts_list:
        fn = tts.ENGINES.get(name)
        if not fn:
            out.append({"engine": name, "ok": False, "error": "unknown engine name"})
            continue
        wav = tmp / f"voicecheck-{user.id}-{name}.wav"
        t = time.time()
        try:
            fn("This is a voice test.", "en-US-GuyNeural", 1.0, wav)
            ok = wav.exists() and wav.stat().st_size > 1000
            out.append({"engine": name, "ok": ok, "seconds": round(time.time() - t, 1), "error": None if ok else "no audio produced"})
        except Exception as e:
            out.append({"engine": name, "ok": False, "seconds": round(time.time() - t, 1), "error": str(e)[:300]})
        finally:
            wav.unlink(missing_ok=True)
    return {"engines": out, "best_quality_available": any(e["engine"] == "edge" and e["ok"] for e in out)}


@router.post("/sadtalker-check")
def sadtalker_check(user: User = Depends(get_current_user)):
    """Really run SadTalker on a short clip on this machine: proves the install works and shows GPU vs CPU speed."""
    _rate_limit(f"sadcheck:{user.id}", 2)
    st = sadtalker.status()
    if not st["ready"]:
        return {"ok": False, "error": "Not installed: " + "; ".join(st["problems"]), "device": None}
    dev = sadtalker.device_info()
    work = storage.subdir("tmp") / f"sadcheck-{user.id}"
    storage.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    try:
        wav = work / "a.wav"
        tts.synthesize("Hello! This is a quick lip sync test.", "en-US-AriaNeural", 1.0, wav)
        t = time.time()
        video = sadtalker.run(catalog.TEST_PORTRAIT, wav, work / "out")
        secs = round(time.time() - t, 1)
        rel = f"tmp/sadcheck-{user.id}/result.mp4"
        Path(storage.root() / rel).write_bytes(video.read_bytes())
        return {"ok": True, "seconds": secs, "device": dev, "video_url": storage.url_for(rel)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:800], "device": dev}
