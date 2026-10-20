"""SadTalker integration, tested against a stand-in with SadTalker's CLI and output layout (the real model needs its
own Python 3.10 env + weights; setup-sadtalker.ps1 runs a real self-test on the user's machine)."""
import sys
import textwrap

import numpy as np
import pytest
from PIL import Image

from app.config import ASSETS_DIR, get_settings
from app.services import avatar as avatar_mod, catalog, face, renderer, sadtalker, tts

FAKE = textwrap.dedent('''
    import argparse, os, subprocess, sys, time
    p = argparse.ArgumentParser()
    for a in ("--driven_audio", "--source_image", "--result_dir", "--checkpoint_dir", "--preprocess", "--size", "--enhancer"):
        p.add_argument(a)
    p.add_argument("--still", action="store_true")
    a = p.parse_args()
    if os.environ.get("FAKE_SADTALKER_FAIL"):
        print("RuntimeError: CUDA out of memory (simulated)", file=sys.stderr); sys.exit(1)
    assert a.still and a.preprocess == "full" and os.path.exists(a.source_image) and os.path.exists(a.driven_audio)
    os.makedirs(a.result_dir, exist_ok=True)
    out = os.path.join(a.result_dir, time.strftime("%Y_%m_%d_%H.%M.%S") + ".mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", a.source_image, "-i", a.driven_audio,
                    "-shortest", "-r", "25", "-pix_fmt", "yuv420p", "-vf", "scale=512:512", out], check=True)
''')


@pytest.fixture()
def fake_sadtalker(tmp_path, monkeypatch):
    d = tmp_path / "SadTalker"
    (d / "checkpoints").mkdir(parents=True)
    (d / "inference.py").write_text(FAKE)
    for f in (*sadtalker.CHECKPOINTS, "SadTalker_V0.0.2_256.safetensors"):
        (d / "checkpoints" / f).write_bytes(b"x")
    s = get_settings()
    monkeypatch.setattr(s, "sadtalker_dir", str(d))
    monkeypatch.setattr(s, "sadtalker_python", sys.executable)
    monkeypatch.setattr(s, "sadtalker_enhancer", "")
    return d


def test_status_reports_missing_pieces(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "sadtalker_dir", "")
    assert sadtalker.status()["ready"] is False
    monkeypatch.setattr(get_settings(), "sadtalker_dir", str(tmp_path))
    st = sadtalker.status()
    assert not st["ready"] and any("inference.py" in p for p in st["problems"]) and any("checkpoint" in p for p in st["problems"])


def _photo_avatar(tmp_path):
    data = pytest.importorskip("skimage.data")
    canvas, meta = face.build_avatar(Image.fromarray(data.astronaut()), True, "female")
    p = tmp_path / "av.png"
    canvas.save(p)
    return p, meta


def test_provider_frames_are_cut_out(fake_sadtalker, tmp_path):
    path, meta = _photo_avatar(tmp_path)
    wav = tmp_path / "a.wav"
    tts.synthesize("Testing the realistic engine", "en-US-AriaNeural", 1.0, wav)
    prov = avatar_mod.get_provider("sadtalker")
    assert prov.name == "sadtalker"
    frames = list(prov.frames(path, meta, wav, 256, 12, 20, tmp_path, seed=0, gestures=False))
    assert len(frames) == 20 and frames[0].size == (256, 256) and frames[0].mode == "RGBA"
    a = np.asarray(frames[5].getchannel("A")).astype(float) / 255
    assert a[:8, :8].mean() < 0.1 and a[100:160, 100:160].mean() > 0.9  # grey background removed, face kept


def test_render_uses_sadtalker_and_falls_back_on_failure(fake_sadtalker, tmp_path, monkeypatch):
    path, meta = _photo_avatar(tmp_path)
    spec = renderer.RenderSpec(scenes=[renderer.SceneSpec("Hello from the neural engine.", {"type": "scenery", "scenery_id": "office"})],
                               avatar_path=path, avatar_meta=meta, animation_engine="sadtalker", intro_enabled=False)
    res = renderer.render_video(spec, tmp_path / "w1", tmp_path / "o1.mp4", tmp_path / "t1.jpg", lambda p, s: None)
    assert res.video.stat().st_size > 5000 and not any("failed" in n for n in res.notes)
    monkeypatch.setenv("FAKE_SADTALKER_FAIL", "1")
    res = renderer.render_video(spec, tmp_path / "w2", tmp_path / "o2.mp4", tmp_path / "t2.jpg", lambda p, s: None)
    assert res.video.stat().st_size > 5000  # still rendered, with the built-in animation
    assert any("sadtalker failed" in n and "CUDA out of memory" in n for n in res.notes)


def test_api_engine_validation(client, auth, fake_sadtalker, monkeypatch):
    pid = client.post("/api/projects", json={}, headers=auth).json()["id"]
    ok = client.put(f"/api/projects/{pid}", json={"animation_engine": "sadtalker"}, headers=auth)
    assert ok.status_code == 200 and ok.json()["animation_engine"] == "sadtalker"
    assert client.get("/api/system").json()["sadtalker"]["ready"] is True
    monkeypatch.setattr(get_settings(), "sadtalker_dir", "")
    bad = client.put(f"/api/projects/{pid}", json={"animation_engine": "sadtalker"}, headers=auth)
    assert bad.status_code == 422 and "not installed" in bad.json()["detail"]
    assert client.put(f"/api/projects/{pid}", json={"animation_engine": "hollywood"}, headers=auth).status_code == 422
