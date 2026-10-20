"""The Colab GPU worker (colab/gpu_worker.py) and the app's remote-GPU engine, tested with stand-in models that follow the
real projects' interfaces (the real EchoMimicV2/SadTalker need a GPU + downloads and run only inside the notebook)."""
import importlib.util
import json
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.services import face, remote_gpu, renderer, tts

WORKER_FILE = Path(__file__).resolve().parents[2] / "colab" / "gpu_worker.py"

FAKE_ECHO = textwrap.dedent('''
    import argparse, os, subprocess, sys, time, datetime
    p = argparse.ArgumentParser()
    p.add_argument("--config"); p.add_argument("-W", type=int); p.add_argument("-H", type=int); p.add_argument("-L", type=int)
    p.add_argument("--fps", type=int)
    for a in ("--ref_images_dir", "--refimg_name", "--audio_dir", "--audio_name", "--pose_dir", "--pose_name"):
        p.add_argument(a)
    a = p.parse_args()
    if os.environ.get("FAKE_ECHO_FAIL"):
        print("torch.OutOfMemoryError: CUDA out of memory (simulated)", file=sys.stderr); sys.exit(1)
    ref = os.path.join(a.ref_images_dir, a.refimg_name); aud = os.path.join(a.audio_dir, a.audio_name)
    assert os.path.exists(ref) and os.path.exists(aud) and os.path.isdir(os.path.join(a.pose_dir, a.pose_name))
    out = os.path.join("output", datetime.date.today().isoformat()); os.makedirs(out, exist_ok=True)
    dst = os.path.join(out, "ref-a-a-i0_sig.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", ref, "-i", aud, "-t", str(a.L / a.fps),
                    "-r", str(a.fps), "-pix_fmt", "yuv420p", "-vf", f"scale={a.W}:{a.H}", dst], check=True)
''')


def load_worker(monkeypatch, tmp_path, stub: bool):
    monkeypatch.setenv("WORKER_TOKEN", "secret")
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("WORKER_STUB", "1" if stub else "0")
    spec = importlib.util.spec_from_file_location(f"gpu_worker_{time.time_ns()}", WORKER_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _inputs(tmp_path, w=360, h=640):
    img = tmp_path / "in.png"
    Image.new("RGB", (w, h), (30, 120, 200)).save(img)
    wav = tmp_path / "in.wav"
    tts.synthesize("Testing the GPU worker", "en-US-GuyNeural", 1.0, wav)
    return img, wav


def _wait(c, job, headers):
    for _ in range(200):
        st = c.get(f"/jobs/{job}", headers=headers).json()
        if st["status"] in ("done", "failed"):
            return st
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)], capture_output=True, text=True).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


def test_worker_protocol_stub(monkeypatch, tmp_path):
    w = load_worker(monkeypatch, tmp_path, stub=True)
    c = TestClient(w.app)
    assert c.get("/health").status_code == 401
    h = {"X-Token": "secret"}
    assert c.get("/health", headers=h).json()["engines"]["echomimic"]["ready"]
    img, wav = _inputs(tmp_path)
    r = c.post("/jobs", headers=h, data={"engine": "echomimic", "fps": "24"},
               files={"image": ("in.png", img.read_bytes(), "image/png"), "audio": ("in.wav", wav.read_bytes(), "audio/wav")})
    st = _wait(c, r.json()["id"], h)
    assert st["status"] == "done", st
    v = c.get(f"/jobs/{r.json()['id']}/video", headers=h)
    assert v.status_code == 200
    (tmp_path / "o.mp4").write_bytes(v.content)
    s = _probe(tmp_path / "o.mp4")
    assert (int(s["video"]["width"]), int(s["video"]["height"])) == (360, 640) and "audio" in s
    assert c.post("/jobs", headers=h, data={"engine": "bogus"}, files={"image": ("i", b"x"), "audio": ("a", b"x")}).status_code == 422


@pytest.fixture()
def fake_models(tmp_path, monkeypatch):
    from tests.test_sadtalker import FAKE as FAKE_SAD

    echo = tmp_path / "echomimic_v2"
    (echo / "configs" / "prompts").mkdir(parents=True)
    (echo / "configs" / "prompts" / "infer_acc.yaml").write_text("x: 1\n")
    (echo / "infer_acc.py").write_text(FAKE_ECHO)
    (echo / "pretrained_weights").mkdir()
    (echo / "pretrained_weights" / "denoising_unet.pth").write_bytes(b"x")
    pose = echo / "assets" / "halfbody_demo" / "pose" / "01"
    pose.mkdir(parents=True)
    for i in range(400):
        np.save(pose / f"{i}.npy", np.zeros(1))
    sad = tmp_path / "SadTalker"
    (sad / "checkpoints").mkdir(parents=True)
    (sad / "inference.py").write_text(FAKE_SAD)
    (sad / "checkpoints" / "SadTalker_V0.0.2_256.safetensors").write_bytes(b"x")
    for k, v in {"ECHOMIMIC_DIR": echo, "SADTALKER_DIR": sad}.items():
        monkeypatch.setenv(k, str(v))
    monkeypatch.setenv("ECHOMIMIC_PY", sys.executable)
    monkeypatch.setenv("SADTALKER_PY", sys.executable)
    return echo, sad


def test_worker_echomimic_and_fallback(monkeypatch, tmp_path, fake_models):
    w = load_worker(monkeypatch, tmp_path, stub=False)
    c = TestClient(w.app)
    h = {"X-Token": "secret"}
    img, wav = _inputs(tmp_path, 400, 700)
    files = lambda: {"image": ("in.png", img.read_bytes(), "image/png"), "audio": ("in.wav", wav.read_bytes(), "audio/wav")}
    st = _wait(c, c.post("/jobs", headers=h, data={"engine": "echomimic", "fps": "24"}, files=files()).json()["id"], h)
    assert st["status"] == "done" and st["engine_used"] == "echomimic", st
    job = [k for k in w.jobs][-1]
    s = _probe(w.WORK / job / "out.mp4")
    assert (int(s["video"]["width"]), int(s["video"]["height"])) == (400, 700)  # square padding cropped back to the input
    monkeypatch.setenv("FAKE_ECHO_FAIL", "1")
    st = _wait(c, c.post("/jobs", headers=h, data={"engine": "echomimic", "fps": "24"}, files=files()).json()["id"], h)
    assert st["status"] == "done" and st["engine_used"] == "sadtalker" and "out of memory" in st["message"], st


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_app_renders_through_remote_worker_in_photo_framing(monkeypatch, tmp_path):
    """App -> HTTP -> worker (stub engine) -> MP4 back -> composited as the original-photo frame."""
    data = pytest.importorskip("skimage.data")
    import uvicorn

    w = load_worker(monkeypatch, tmp_path, stub=True)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(w.app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        remote_gpu.save(f"http://127.0.0.1:{port}", "secret")
        assert remote_gpu.health()["ok"]
        canvas, meta, full = face.build_avatar_full(Image.fromarray(data.astronaut()), True, "female")
        cpath, fpath = tmp_path / "c.png", tmp_path / "f.jpg"
        canvas.save(cpath)
        full.save(fpath)
        spec = renderer.RenderSpec(scenes=[renderer.SceneSpec("Hello from the GPU worker.", {"type": "gradient"})],
                                   aspect_ratio="9:16", avatar_path=cpath, avatar_meta=meta, animation_engine="gpu_echomimic",
                                   framing="photo", photo_full_path=fpath, intro_enabled=False, captions_enabled=False)
        res = renderer.render_video(spec, tmp_path / "w", tmp_path / "o.mp4", tmp_path / "t.jpg", lambda p, s: None)
        assert res.video.stat().st_size > 5000 and any("stub echomimic" in n for n in res.notes), res.notes
        frame = np.asarray(Image.open(res.thumbnail).convert("RGB")).astype(int)
        assert frame.mean() > 40  # the worker's frames fill the picture (no black, no cut-out holes)
        remote_gpu.save("http://127.0.0.1:9", "secret")  # nothing listens there: falls back to the built-in engine
        res = renderer.render_video(spec, tmp_path / "w2", tmp_path / "o2.mp4", tmp_path / "t2.jpg", lambda p, s: None)
        assert res.video.exists() and any("gpu_echomimic failed" in n for n in res.notes), res.notes
    finally:
        remote_gpu.clear()
        server.should_exit = True


def test_api_remote_gpu_settings(client, auth):
    r = client.put("/api/system/remote-gpu", headers=auth, json={"url": "http://127.0.0.1:9", "token": "t"})
    assert r.status_code == 200 and r.json()["ok"] is False  # saved, but not reachable
    assert client.get("/api/system").json()["remote_gpu"]["configured"] is True
    pid = client.post("/api/projects", json={}, headers=auth).json()["id"]
    ok = client.put(f"/api/projects/{pid}", headers=auth, json={"animation_engine": "gpu_echomimic", "framing": "photo"})
    assert ok.status_code == 200 and ok.json()["animation_engine"] == "gpu_echomimic" and ok.json()["framing"] == "photo"
    assert client.delete("/api/system/remote-gpu", headers=auth).status_code == 204
    bad = client.put(f"/api/projects/{pid}", headers=auth, json={"animation_engine": "gpu_sadtalker"})
    assert bad.status_code == 422 and "Colab" in bad.json()["detail"]
