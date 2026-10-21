"""Client for a remote GPU worker (the VideoAI Colab notebook, or any server speaking the same small protocol):

    GET  /health                -> {"ok": true, "gpu": "...", "engines": {"echomimic": {...}, "sadtalker": {...}}}
    POST /jobs  (multipart)     -> {"id": "..."}       fields: engine, fps; files: image (or video for "lipsync"), audio
    GET  /jobs/{id}             -> {"status": "queued|running|done|failed", "message": "...", "engine_used": "..."}
    GET  /jobs/{id}/video       -> video/mp4
Every request carries the header X-Token. The URL/token are saved in storage/remote_gpu.json (set from System check)."""
import json
import time
from pathlib import Path

import httpx

from . import storage


class RemoteGPUError(RuntimeError):
    pass


def _file() -> Path:
    return storage.root() / "remote_gpu.json"


def get() -> dict:
    try:
        return json.loads(_file().read_text())
    except (OSError, ValueError):
        return {}


def save(url: str, token: str) -> dict:
    data = {"url": url.strip().rstrip("/"), "token": token.strip()}
    _file().write_text(json.dumps(data))
    return data


def clear() -> None:
    _file().unlink(missing_ok=True)


def configured() -> bool:
    return bool(get().get("url"))


def _client(timeout: float = 60) -> tuple[httpx.Client, str]:
    cfg = get()
    if not cfg.get("url"):
        raise RemoteGPUError("No GPU worker connected: start the Colab notebook and paste its URL on the System check page")
    return httpx.Client(timeout=timeout, headers={"X-Token": cfg.get("token", "")}, follow_redirects=True), cfg["url"]


def health() -> dict:
    client, url = _client(20)
    with client:
        try:
            r = client.get(f"{url}/health")
        except httpx.HTTPError as e:
            raise RemoteGPUError(f"GPU worker not reachable ({e.__class__.__name__}). Is the Colab notebook still running?")
        if r.status_code == 401:
            raise RemoteGPUError("GPU worker rejected the token: copy the token shown in the notebook again")
        r.raise_for_status()
        return r.json()


def render(engine: str, image: Path, audio: Path, out: Path, fps: int, timeout_s: int = 3 * 3600, on_status=None) -> tuple[Path, str]:
    """Run one clip on the worker; returns (mp4 path, message from the worker, e.g. which engine actually ran).
    `image` is a PNG, or for engine "lipsync" an MP4 whose lower face gets re-generated to match the audio."""
    client, url = _client(120)
    field, ctype = ("video", "video/mp4") if image.suffix.lower() == ".mp4" else ("image", "image/png")
    with client:
        try:
            with open(image, "rb") as fi, open(audio, "rb") as fa:
                r = client.post(f"{url}/jobs", data={"engine": engine, "fps": str(fps)},
                                files={field: (image.name, fi, ctype), "audio": (audio.name, fa, "audio/wav")})
        except httpx.HTTPError as e:
            raise RemoteGPUError(f"could not send the job to the GPU worker ({e.__class__.__name__})")
        if r.status_code >= 400:
            raise RemoteGPUError(f"GPU worker refused the job: {r.text[:300]}")
        job = r.json()["id"]
        t0, last = time.time(), {}
        while True:
            if time.time() - t0 > timeout_s:
                raise RemoteGPUError(f"GPU worker took longer than {timeout_s // 60} minutes")
            try:
                last = client.get(f"{url}/jobs/{job}").json()
            except (httpx.HTTPError, ValueError):
                last = {"status": "unknown"}  # transient tunnel hiccup: keep polling
            if on_status:
                on_status(last)
            if last.get("status") == "done":
                break
            if last.get("status") == "failed":
                raise RemoteGPUError("GPU worker failed: " + str(last.get("message", ""))[:500])
            time.sleep(5)
        with client.stream("GET", f"{url}/jobs/{job}/video") as resp:
            if resp.status_code >= 400:
                raise RemoteGPUError(f"could not download the result ({resp.status_code})")
            with open(out, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
        return out, str(last.get("message", ""))
