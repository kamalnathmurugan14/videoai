"""VideoAI GPU worker: runs neural talking-avatar engines on a GPU (Google Colab) for the VideoAI app.

Protocol (all requests need header X-Token):
    GET  /health             engines + GPU info
    POST /jobs               multipart: engine=echomimic|sadtalker, fps, image (PNG), audio (WAV)  -> {"id"}
    GET  /jobs/{id}          {"status": queued|running|done|failed, "message", "engine_used"}
    GET  /jobs/{id}/video    the MP4 (same size as the input image, with the input audio)

Engines (installed by the notebook, each in its own Python 3.10 virtualenv):
    echomimic -> EchoMimicV2 (https://github.com/antgroup/echomimic_v2, Apache-2.0): half body + hand gestures
    sadtalker -> SadTalker  (https://github.com/OpenTalker/SadTalker, Apache-2.0): face / head
If EchoMimicV2 fails (e.g. out of GPU memory) the job automatically falls back to SadTalker and says so.

Environment: WORKER_TOKEN, WORK_DIR, ECHOMIMIC_DIR, ECHOMIMIC_PY, SADTALKER_DIR, SADTALKER_PY,
             WORKER_STUB=1 (testing only: "engines" just turn the image + audio into a video with ffmpeg).
"""
import os
import queue
import re
import secrets
import shutil
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

TOKEN = os.environ.get("WORKER_TOKEN") or secrets.token_urlsafe(16)
WORK = Path(os.environ.get("WORK_DIR", "/content/videoai_jobs"))
WORK.mkdir(parents=True, exist_ok=True)
STUB = os.environ.get("WORKER_STUB") == "1"
ENG = {
    "echomimic": (Path(os.environ.get("ECHOMIMIC_DIR", "/content/echomimic_v2")), os.environ.get("ECHOMIMIC_PY", "/content/env_echomimic/bin/python")),
    "sadtalker": (Path(os.environ.get("SADTALKER_DIR", "/content/SadTalker")), os.environ.get("SADTALKER_PY", "/content/env_sadtalker/bin/python")),
}
app = FastAPI(title="VideoAI GPU worker")
jobs: dict[str, dict] = {}
q: "queue.Queue[str]" = queue.Queue()


class EngineError(RuntimeError):
    pass


def _auth(token: str | None) -> None:
    if not token or not secrets.compare_digest(token, TOKEN):
        raise HTTPException(401, "bad token")


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 4 * 3600) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-8:]
        raise EngineError(" | ".join(tail)[-700:] or f"exit code {r.returncode}")
    return r


def _ffmpeg(*args: str) -> None:
    _run(["ffmpeg", "-y", "-loglevel", "error", *args])


def _image_size(path: Path) -> tuple[int, int]:
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)]).stdout
    w, h = out.strip().split(",")[:2]
    return int(w), int(h)


def _audio_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def _finish(video: Path, image: Path, audio: Path, out: Path, fps: int, crop: str | None = None) -> Path:
    """Normalise any engine output: optional crop, scale to the input image size, given fps, the job's own audio."""
    w, h = _image_size(image)
    w, h = w - w % 2, h - h % 2
    vf = (crop + "," if crop else "") + f"scale={w}:{h},fps={fps}"
    _ffmpeg("-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-vf", vf, "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out))
    return out


def engine_status(name: str) -> dict:
    if STUB:
        return {"ready": True, "stub": True}
    d, py = ENG[name]
    problems = []
    if not Path(py).exists():
        problems.append(f"python env missing ({py})")
    if name == "sadtalker":
        if not (d / "inference.py").exists():
            problems.append("code missing")
        if not (d / "checkpoints" / "SadTalker_V0.0.2_256.safetensors").exists():
            problems.append("weights missing")
    else:
        if not any((d / s).exists() for s in ("infer_acc.py", "infer.py")):
            problems.append("code missing")
        if not (d / "pretrained_weights" / "denoising_unet.pth").exists():
            problems.append("weights missing")
    return {"ready": not problems, "problems": problems}


# ----------------------------------------------------------------------------- engines
def run_stub(image: Path, audio: Path, job: Path) -> Path:
    raw = job / "stub.mp4"
    _ffmpeg("-loop", "1", "-i", str(image), "-i", str(audio), "-shortest", "-r", "25", "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(raw))
    return raw


def run_sadtalker(image: Path, audio: Path, job: Path) -> Path:
    d, py = ENG["sadtalker"]
    out = job / "sadtalker"
    _run([py, "inference.py", "--driven_audio", str(audio), "--source_image", str(image), "--result_dir", str(out),
          "--still", "--preprocess", "full", "--enhancer", "gfpgan"], cwd=d)
    vids = sorted(out.glob("*.mp4"), key=lambda p: p.stat().st_mtime) or sorted(out.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not vids:
        raise EngineError("SadTalker produced no video")
    return vids[-1]


ECHO_SIZE = 768


def run_echomimic(image: Path, audio: Path, job: Path, fps: int) -> tuple[Path, str]:
    """EchoMimicV2 needs a square reference image and a pose sequence (it provides demo poses, which drive the hands)."""
    d, py = ENG["echomimic"]
    script = "infer_acc.py" if (d / "infer_acc.py").exists() else "infer.py"
    config = d / "configs" / "prompts" / ("infer_acc.yaml" if script == "infer_acc.py" else "infer.yaml")
    helptext = _run([py, script, "--help"], cwd=d, timeout=600).stdout
    has = lambda flag: re.search(rf"(^|\s){re.escape(flag)}[\s,=]", helptext) is not None
    needed = ["--ref_images_dir", "--refimg_name", "--audio_dir", "--audio_name", "--pose_dir", "--pose_name"]
    missing = [f for f in needed if not has(f)]
    if missing:
        raise EngineError(f"EchoMimicV2's {script} has a different interface than expected (missing {', '.join(missing)})")
    # square, edge-extended reference so the person keeps their framing; cropped back afterwards
    w, h = _image_size(image)
    side = max(w, h)
    ref_dir, aud_dir = job / "ref", job / "aud"
    ref_dir.mkdir(exist_ok=True)
    aud_dir.mkdir(exist_ok=True)
    pad_x, pad_y = (side - w) // 2, (side - h) // 2
    _ffmpeg("-i", str(image), "-vf", f"pad={side}:{side}:{pad_x}:{pad_y}:color=0x808080,scale={ECHO_SIZE}:{ECHO_SIZE}", str(ref_dir / "ref.png"))
    _ffmpeg("-i", str(audio), "-ar", "16000", "-ac", "1", str(aud_dir / "a.wav"))
    pose_root = d / "assets" / "halfbody_demo" / "pose"
    poses = sorted(p for p in pose_root.iterdir() if p.is_dir() and any(p.glob("*.npy"))) if pose_root.exists() else []
    if not poses:
        raise EngineError(f"no pose sequences found in {pose_root}")
    pose = poses[int(time.time()) % len(poses)]  # vary gestures between clips
    n_pose = len(list(pose.glob("*.npy")))
    n_need = int(_audio_seconds(audio) * 24) + 1
    if n_need > n_pose:
        raise EngineError(f"clip is {n_need / 24:.1f}s but the pose sequence only has {n_pose / 24:.1f}s; split the scene")
    args = [py, script]
    if has("--config"):
        args += ["--config", str(config)]
    for flag, val in (("-W", ECHO_SIZE), ("-H", ECHO_SIZE), ("-L", n_need), ("--fps", 24)):
        if has(flag):
            args += [flag, str(val)]
    args += ["--ref_images_dir", str(ref_dir), "--refimg_name", "ref.png", "--audio_dir", str(aud_dir), "--audio_name", "a.wav",
             "--pose_dir", str(pose_root), "--pose_name", pose.name]
    t0 = time.time()
    _run(args, cwd=d)
    vids = [p for p in (d / "output").rglob("*.mp4") if p.stat().st_mtime >= t0 - 1]
    vids.sort(key=lambda p: ("sig" in p.name, p.stat().st_mtime))
    if not vids:
        raise EngineError("EchoMimicV2 produced no video")
    # undo the square padding: crop the original aspect back out of the square output
    crop = f"crop=iw*{w}/{side}:ih*{h}/{side}:iw*{pad_x}/{side}:ih*{pad_y}/{side}"
    return vids[-1], crop


def process(job_id: str) -> None:
    j = jobs[job_id]
    jd = WORK / job_id
    j["status"] = "running"
    t0 = time.time()
    try:
        image, audio, out = jd / "in.png", jd / "in.wav", jd / "out.mp4"
        if STUB:
            _finish(run_stub(image, audio, jd), image, audio, out, j["fps"])
            j["engine_used"], j["message"] = j["engine"], f"stub {j['engine']}"
        elif j["engine"] == "echomimic":
            try:
                raw, crop = run_echomimic(image, audio, jd, j["fps"])
                _finish(raw, image, audio, out, j["fps"], crop)
                j["engine_used"], j["message"] = "echomimic", f"EchoMimicV2 on the GPU in {time.time() - t0:.0f}s"
            except Exception as e:  # out of memory, interface change, missing weights ...
                raw = run_sadtalker(image, audio, jd)
                _finish(raw, image, audio, out, j["fps"])
                j["engine_used"] = "sadtalker"
                j["message"] = f"EchoMimicV2 failed ({str(e)[:300]}); used SadTalker on the GPU instead"
        else:
            _finish(run_sadtalker(image, audio, jd), image, audio, out, j["fps"])
            j["engine_used"], j["message"] = "sadtalker", f"SadTalker on the GPU in {time.time() - t0:.0f}s"
        j["status"] = "done"
    except Exception as e:
        j["status"], j["message"] = "failed", str(e)[:800]
    j["seconds"] = round(time.time() - t0, 1)


def _loop() -> None:
    while True:
        process(q.get())


threading.Thread(target=_loop, daemon=True).start()


@app.get("/health")
def health(x_token: str | None = Header(default=None)):
    _auth(x_token)
    gpu = ""
    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {"ok": True, "gpu": gpu or ("stub" if STUB else "none"), "queued": q.qsize(),
            "engines": {k: engine_status(k) for k in ENG}}


@app.post("/jobs")
async def create_job(engine: str = Form(...), fps: int = Form(24), image: UploadFile = File(...), audio: UploadFile = File(...),
                     x_token: str | None = Header(default=None)):
    _auth(x_token)
    if engine not in ENG:
        raise HTTPException(422, "engine must be echomimic or sadtalker")
    st = engine_status(engine)
    if not st["ready"] and not (engine == "echomimic" and engine_status("sadtalker")["ready"]):
        raise HTTPException(503, f"{engine} is not installed: {st.get('problems')}")
    job_id = uuid.uuid4().hex[:12]
    jd = WORK / job_id
    jd.mkdir(parents=True)
    (jd / "in.png").write_bytes(await image.read())
    (jd / "in.wav").write_bytes(await audio.read())
    jobs[job_id] = {"status": "queued", "engine": engine, "fps": max(1, min(60, fps)), "message": "", "engine_used": None}
    q.put(job_id)
    return {"id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str, x_token: str | None = Header(default=None)):
    _auth(x_token)
    if job_id not in jobs:
        raise HTTPException(404, "unknown job")
    return jobs[job_id]


@app.get("/jobs/{job_id}/video")
def job_video(job_id: str, x_token: str | None = Header(default=None)):
    _auth(x_token)
    out = WORK / job_id / "out.mp4"
    if jobs.get(job_id, {}).get("status") != "done" or not out.exists():
        raise HTTPException(404, "not ready")
    return FileResponse(out, media_type="video/mp4")
