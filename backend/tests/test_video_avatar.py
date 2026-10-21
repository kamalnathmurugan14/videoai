"""Video avatars: upload a short clip of a person, loop it with a re-animated mouth, and lip-sync it on the GPU worker
(tested with a stand-in LatentSync that follows the real project's command-line interface)."""
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import ASSETS_DIR
from app.services import remote_gpu, renderer, storage, tts, video_avatar
from tests.test_gpu_worker import _probe, load_worker

FAKE_LATENTSYNC = textwrap.dedent('''
    import argparse, subprocess
    p = argparse.ArgumentParser()
    for a in ("--unet_config_path", "--inference_ckpt_path", "--video_path", "--audio_path", "--video_out_path",
              "--inference_steps", "--guidance_scale", "--seed"):
        p.add_argument(a)
    p.add_argument("--enable_deepcache", action="store_true")
    a = p.parse_args()
    # mark the frames (a red square) so the test can see the result came back from the "model"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.video_path, "-i", a.audio_path, "-vf",
                    "drawbox=x=0:y=0:w=24:h=24:color=red:t=fill", "-shortest", a.video_out_path], check=True)
''')


def make_clip(path, seconds=3.0, face=True):
    """A short 'selfie video': the test portrait gently rocking and drifting (or a plain grey clip without a face)."""
    src = ["-loop", "1", "-i", str(ASSETS_DIR / "test" / "portrait.jpg")] if face else ["-f", "lavfi", "-i", "color=gray:s=480x640"]
    vf = "scale=480:640,rotate=0.04*sin(2*PI*t):fillcolor=gray,crop=440:600:20+10*sin(PI*t):20" if face else "null"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src, "-t", str(seconds), "-r", "30", "-vf", vf,
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    return make_clip(tmp_path_factory.mktemp("clip") / "me.mp4")


def test_upload_video_avatar_and_delete(client, auth, clip, tmp_path):
    r = client.post("/api/media", headers=auth, data={"kind": "avatar_video", "gender": "male"},
                    files={"file": ("me.mp4", clip.read_bytes(), "video/mp4")})
    assert r.status_code == 201, r.text
    a = r.json()
    assert a["kind"] == "avatar" and a["video_url"] and 2.5 < a["video_seconds"] < 3.5 and a["face_found"]
    assert client.get(a["video_url"]).status_code == 200
    up = storage.subdir("uploads")
    assert len(list((up / f"{a['id']}_frames").glob("*.jpg"))) == 75  # 3 s at 25 fps
    with np.load(up / f"{a['id']}_track.npz") as z:
        assert z["found"].mean() > 0.9 and z["upper"].shape == (75, 11, 2)
    # bad uploads are rejected with a clear message
    bad = client.post("/api/media", headers=auth, data={"kind": "avatar_video"}, files={"file": ("x.mp4", b"not a video", "video/mp4")})
    assert bad.status_code == 400
    grey = make_clip(tmp_path / "grey.mp4", face=False)
    r2 = client.post("/api/media", headers=auth, data={"kind": "avatar_video"}, files={"file": ("g.mp4", grey.read_bytes(), "video/mp4")})
    assert r2.status_code == 400 and "face" in r2.json()["detail"]
    assert not any(up.glob("*.upload")) and not any(storage.subdir("tmp").glob("*.upload"))
    assert client.delete(f"/api/media/{a['id']}", headers=auth).status_code == 204
    assert not (up / f"{a['id']}_frames").exists() and not (up / f"{a['id']}_track.npz").exists()


def _ingest(clip):
    _, _, _, meta = video_avatar.ingest(clip.read_bytes(), "female")
    return meta


def test_mouth_animation_follows_the_tracked_face(clip, tmp_path):
    av = video_avatar.load(_ingest(clip))
    assert [av.index_at(t) for t in (0, 1.0, av.n / 25, (av.n - 1) / 25 + 1.0)] == [0, 25, av.n - 2, av.n - 1 - 25]
    i = 30
    import cv2
    rgb = cv2.cvtColor(cv2.imread(str(av.files[i])), cv2.COLOR_BGR2RGB)
    closed, opened = av.animate(rgb, i, 0.0), av.animate(rgb, i, 0.9)
    up, lo = av.tr["upper"][i], av.tr["lower"][i]
    x0, x1 = int(up[:, 0].min()), int(up[:, 0].max())
    y = int((up[5, 1] + lo[5, 1]) / 2)
    fw = float(av.tr["face_w"][i])
    mouth = (slice(y - 2, y + int(0.1 * fw)), slice(x0 + 4, x1 - 4))
    assert opened[mouth].mean() < closed[mouth].mean() - 25  # a dark opening appears exactly at the lips
    far = (slice(0, int(up[:, 1].min() - fw * 0.5)), slice(None))
    assert np.array_equal(opened[far], rgb[far])  # nothing above the mouth (eyes, forehead) is touched


def test_render_with_video_avatar(clip, tmp_path):
    meta = _ingest(clip)
    spec = renderer.RenderSpec(scenes=[renderer.SceneSpec("Hello from my own video.", {"type": "color", "color": "#000000"}),
                                       renderer.SceneSpec("A second scene continues the motion.", {"type": "color"})],
                               aspect_ratio="9:16", avatar_path=storage.subdir("uploads") / f"{meta['video']['frames'][:-7]}.png",
                               avatar_meta=meta, video_avatar=meta, intro_enabled=False, captions_enabled=False, gestures_enabled=True)
    out = tmp_path / "v.mp4"
    res = renderer.render_video(spec, tmp_path / "w", out, tmp_path / "t.jpg", lambda *_: None)
    assert out.exists() and res.duration > 2 and not any(n.startswith("⚠") for n in res.notes), res.notes
    w, h = renderer.dims("9:16")
    frames = list(renderer._decode(out, w, h, 12, 30))
    a = [np.asarray(f, np.float32) for f in frames]
    assert np.abs(a[0] - a[-1]).mean() > 1.0  # the real footage moves
    assert float(np.asarray(frames[5])[: h // 10].mean()) > 60  # the clip fills the frame (no black background)


def test_gpu_lipsync_through_worker_and_fallback(clip, tmp_path, monkeypatch):
    ls = tmp_path / "LatentSync"
    (ls / "scripts").mkdir(parents=True)
    (ls / "scripts" / "inference.py").write_text(FAKE_LATENTSYNC)
    (ls / "configs" / "unet").mkdir(parents=True)
    (ls / "configs" / "unet" / "stage2.yaml").write_text("x: 1\n")
    (ls / "checkpoints" / "whisper").mkdir(parents=True)
    (ls / "checkpoints" / "latentsync_unet.pt").write_bytes(b"x")
    (ls / "checkpoints" / "whisper" / "tiny.pt").write_bytes(b"x")
    monkeypatch.setenv("LATENTSYNC_DIR", str(ls))
    monkeypatch.setenv("LATENTSYNC_PY", sys.executable)
    w = load_worker(monkeypatch, tmp_path, stub=False)
    assert w.engine_status("lipsync")["ready"]
    monkeypatch.setattr(remote_gpu, "_client", lambda timeout=60: (TestClient(w.app, headers={"X-Token": "secret"}), "http://testserver"))
    monkeypatch.setattr(remote_gpu.time, "sleep", lambda s: None)
    meta = _ingest(clip)
    spec = renderer.RenderSpec(scenes=[renderer.SceneSpec("Lip sync on the GPU.", {"type": "color"})], aspect_ratio="9:16",
                               avatar_path=storage.subdir("uploads") / f"{meta['video']['frames'][:-7]}.png", avatar_meta=meta,
                               video_avatar=meta, animation_engine="gpu_lipsync", captions_enabled=False, intro_enabled=True,
                               intro_title="Hi")
    out = tmp_path / "ls.mp4"
    res = renderer.render_video(spec, tmp_path / "w1", out, tmp_path / "t.jpg", lambda *_: None)
    assert any("LatentSync" in n for n in res.notes) and not any(n.startswith("⚠") for n in res.notes), res.notes
    wd, hd = renderer.dims("9:16")
    f = np.asarray(next(renderer._decode(out, wd, hd, 12, 1)), np.int32)
    assert f[2, 2, 0] > 180 and f[2, 2, 1] < 90  # the model's output (red mark) is in the final video
    s = _probe(out)
    assert "audio" in s and (int(s["video"]["width"]), int(s["video"]["height"])) == (wd, hd)

    # worker down -> the built-in mouth animation is used, with a note
    def down(*a, **k):
        raise remote_gpu.RemoteGPUError("GPU worker not reachable")
    monkeypatch.setattr(remote_gpu, "render", down)
    res2 = renderer.render_video(spec, tmp_path / "w2", tmp_path / "fb.mp4", tmp_path / "t2.jpg", lambda *_: None)
    assert any("GPU lip-sync failed" in n for n in res2.notes) and (tmp_path / "fb.mp4").exists()


def test_gpu_lipsync_for_photos_uses_mouthless_motion(tmp_path, monkeypatch):
    """With a photo avatar the built-in animation supplies head motion without a drawn mouth; the GPU adds the mouth."""
    sent = {}

    def fake_render(engine, video, audio, out, fps, **k):
        sent["engine"], sent["suffix"] = engine, video.suffix
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio), "-shortest", str(out)], check=True)
        return out, "LatentSync lip-sync on the GPU in 1s"
    monkeypatch.setattr(remote_gpu, "render", fake_render)
    from app.services import catalog
    meta = catalog.load_avatars()[0]
    av = ASSETS_DIR / "avatars" / meta["image"]
    spec = renderer.RenderSpec(scenes=[renderer.SceneSpec("Photo plus GPU lip sync.", {"type": "color"})], avatar_path=av,
                               avatar_meta=meta, animation_engine="gpu_lipsync", captions_enabled=False)
    res = renderer.render_video(spec, tmp_path / "w", tmp_path / "p.mp4", tmp_path / "t.jpg", lambda *_: None)
    assert sent == {"engine": "lipsync", "suffix": ".mp4"} and (tmp_path / "p.mp4").exists(), res.notes
