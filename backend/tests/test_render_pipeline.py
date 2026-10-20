import json
import subprocess
import wave

import pytest

from app.config import ASSETS_DIR, get_settings
from app.services import avatar as avatar_mod, captions, catalog, renderer, tts


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)], capture_output=True, text=True).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


def test_caption_timing_and_chunks():
    words, joiner = captions.split_words("Hello there, this is a longer caption test sentence. And another one!")
    t = captions.word_timings(words, 0.2, 4.2)
    assert t[0][0] == pytest.approx(0.2) and t[-1][1] == pytest.approx(4.2)
    assert all(a[1] <= b[0] + 1e-6 for a, b in zip(t, t[1:]))
    chunks = captions.chunk_words(words)
    assert [i for c in chunks for i in c] == list(range(len(words)))


def test_tts_and_amplitude(tmp_path):
    wav = tmp_path / "a.wav"
    assert tts.synthesize("Testing one two three", "en-US-AriaNeural", 1.0, wav) == ("synthetic", [])
    env = avatar_mod.amplitude_envelope(wav, 12, 20)
    assert env.max() > 0.5 and env.min() >= 0


def test_render_smoke_with_music_and_media(tmp_path):
    from PIL import Image

    media = tmp_path / "card.png"
    Image.new("RGB", (400, 300), "#22c55e").save(media)
    av = catalog.load_avatars()[0]
    spec = renderer.RenderSpec(
        scenes=[
            renderer.SceneSpec("Hello! Welcome to the smoke test.", {"type": "gradient"}),
            renderer.SceneSpec("Second scene with an image.", {"type": "color", "color": "#112233"}, media),
        ],
        aspect_ratio="9:16", avatar_path=ASSETS_DIR / "avatars" / av["image"], avatar_meta=av,
    )
    from app.services import music

    music.ensure_library()
    spec.music_path = music.path_for("calm")
    stages = []
    res = renderer.render_video(spec, tmp_path / "w", tmp_path / "out.mp4", tmp_path / "t.jpg", lambda p, s: stages.append(p))
    assert res.video.stat().st_size > 5000 and res.thumbnail.exists()
    st = probe(res.video)
    assert st["video"]["codec_name"] == "h264" and st["audio"]["codec_name"] == "aac"
    w, h = renderer.dims("9:16")
    assert (int(st["video"]["width"]), int(st["video"]["height"])) == (w, h)
    assert res.duration > 2.0
    assert stages[-1] == 100 and stages == sorted(stages)


def test_worker_end_to_end(client, auth):
    """API -> queue -> worker -> MP4 on disk, served via /storage and /download."""
    from app.workers import worker

    pid = client.post("/api/projects", json={"template_id": "explainer"}, headers=auth).json()["id"]
    client.put(f"/api/projects/{pid}", headers=auth, json={"scenes": [{"text": "One short scene."}], "aspect_ratio": "1:1"})
    job = client.post(f"/api/projects/{pid}/render", headers=auth).json()
    assert job["status"] == "queued"
    assert client.get(f"/api/projects/{pid}", headers=auth).json()["status"] == "queued"
    assert worker.process_next() is True
    j = client.get(f"/api/jobs/{job['id']}", headers=auth).json()
    assert j["status"] == "done" and j["progress"] == 100, j
    p = client.get(f"/api/projects/{pid}", headers=auth).json()
    assert p["status"] == "done" and p["video_url"] and p["thumbnail_url"] and p["duration"] > 1
    assert client.get(p["video_url"]).headers["content-type"] == "video/mp4"
    dl = client.get(f"/api/projects/{pid}/download", headers=auth)
    assert dl.status_code == 200 and "attachment" in dl.headers["content-disposition"]
    assert worker.process_next() is False


def test_failed_render_reports_error(client, auth, monkeypatch):
    from app.workers import worker

    monkeypatch.setattr(get_settings(), "tts_providers", "espeak-nonexistent")
    pid = client.post("/api/projects", json={}, headers=auth).json()["id"]
    client.put(f"/api/projects/{pid}", headers=auth, json={"scenes": [{"text": "This will fail."}]})
    job = client.post(f"/api/projects/{pid}/render", headers=auth).json()
    worker.process_next()
    j = client.get(f"/api/jobs/{job['id']}", headers=auth).json()
    assert j["status"] == "failed" and "speech" in j["error"].lower()
    assert client.get(f"/api/projects/{pid}", headers=auth).json()["status"] == "failed"


def test_command_avatar_provider(tmp_path, monkeypatch):
    """Pluggable lip-sync provider: a fake 'model' command that just makes a video from image+audio."""
    av = catalog.load_avatars()[0]
    wav = tmp_path / "a.wav"
    tts.synthesize("Plugin provider test", "en-US-AriaNeural", 1.0, wav)
    cmd = "ffmpeg -y -loglevel error -loop 1 -i {image} -i {audio} -t 1 -pix_fmt yuv420p {out}"
    monkeypatch.setattr(get_settings(), "avatar_command", cmd)
    monkeypatch.setattr(get_settings(), "avatar_provider", "command")
    p = avatar_mod.get_provider()
    assert p.name == "command"
    frames = list(p.frames(ASSETS_DIR / "avatars" / av["image"], av, wav, 128, 12, 6, tmp_path))
    assert len(frames) == 6 and frames[0].size == (128, 128)


def test_photo_avatar_pipeline(tmp_path):
    """Photo -> normalised avatar canvas with landmarks -> puppet frames (jaw/eyes/head motion) for a face-less image too."""
    import numpy as np
    from PIL import Image

    from app.services import face

    rng = np.random.default_rng(0)
    photo = Image.fromarray((rng.random((300, 240, 3)) * 255).astype("uint8"))  # no detectable face -> heuristic landmarks
    canvas, meta = face.build_avatar(photo, True, "male")
    assert canvas.size == (face.CANVAS, face.CANVAS) and meta["found"] is False and meta["gender"] == "male"
    assert 0 < meta["mouth"][1] < 1 and len(meta["eyes"]) == 2 and meta["has_mouth_art"] is False
    path = tmp_path / "av.png"
    canvas.save(path)
    wav = tmp_path / "a.wav"
    tts.synthesize("Moving head test sentence for the puppet", "en-US-GuyNeural", 1.0, wav)
    frames = list(avatar_mod.PuppetAvatarProvider().frames(path, meta, wav, 160, 12, 24, tmp_path, seed=2))
    assert len(frames) == 24 and frames[0].size == (160, 160)
    import numpy as np
    a, b = np.asarray(frames[0]).astype(int), np.asarray(frames[12]).astype(int)
    assert np.abs(a - b).sum() > 0  # frames actually differ (motion)


def test_scenery_intro_and_crossfade_render(tmp_path):
    av = catalog.load_avatars()[3]
    spec = renderer.RenderSpec(
        scenes=[
            renderer.SceneSpec("First scene on the office.", {"type": "scenery", "scenery_id": "office"}),
            renderer.SceneSpec("Second scene by the beach.", {"type": "scenery", "scenery_id": "beach"}),
        ],
        aspect_ratio="16:9", avatar_path=ASSETS_DIR / "avatars" / av["image"], avatar_meta=av,
        intro_enabled=True, intro_title="My Intro Title",
    )
    res = renderer.render_video(spec, tmp_path / "w", tmp_path / "o.mp4", tmp_path / "t.jpg", lambda p, s: None)
    assert res.duration > 2.3 + 1.6  # intro (2.3 s) + two scenes
    assert res.thumbnail.exists()


def test_voice_gender_and_engine_notes(tmp_path, monkeypatch):
    assert tts.voice_gender("en-US-GuyNeural") == "male" and tts.voice_gender("en-US-AriaNeural") == "female"
    monkeypatch.setattr(get_settings(), "tts_providers", "gtts-missing,synthetic")
    engine, errors = tts.synthesize("Fallback reporting", "en-US-GuyNeural", 1.0, tmp_path / "x.wav")
    assert engine == "synthetic" and errors == []  # unknown engine names are skipped silently
    monkeypatch.setattr(tts, "ENGINES", {**tts.ENGINES, "edge": lambda *a: (_ for _ in ()).throw(RuntimeError("no network"))})
    monkeypatch.setattr(get_settings(), "tts_providers", "edge,synthetic")
    engine, errors = tts.synthesize("Fallback reporting", "en-US-GuyNeural", 1.0, tmp_path / "y.wav")
    assert engine == "synthetic" and "edge: no network" in errors[0]
