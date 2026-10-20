import io

from PIL import Image


def test_catalogs(client):
    assert len(client.get("/api/avatars").json()) >= 5
    assert len(client.get("/api/voices").json()) > 10
    assert {t["id"] for t in client.get("/api/templates").json()} >= {"explainer", "social-promo", "product-intro"}
    assert client.get("/health").status_code in (200, 404)
    assert client.get("/api/health").json() == {"status": "ok"}


def test_crud(client, auth):
    r = client.post("/api/projects", json={"title": "My video", "template_id": "social-promo"}, headers=auth)
    assert r.status_code == 201
    p = r.json()
    assert p["status"] == "draft" and p["aspect_ratio"] == "9:16" and len(p["scenes"]) == 3

    r = client.put(f"/api/projects/{p['id']}", headers=auth, json={
        "title": "Renamed", "aspect_ratio": "1:1", "voice": "en-GB-RyanNeural", "speed": 1.2,
        "caption_style": "classic", "music_id": "calm",
        "background": {"type": "color", "color": "#112233"},
        "scenes": [{"text": "Hello there."}, {"text": "Second scene.", "background": {"type": "gradient"}}],
    })
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["title"] == "Renamed" and out["aspect_ratio"] == "1:1" and len(out["scenes"]) == 2
    assert out["background"]["color"] == "#112233" and out["music_id"] == "calm"

    assert [x["id"] for x in client.get("/api/projects", headers=auth).json()] == [p["id"]]
    dup = client.post(f"/api/projects/{p['id']}/duplicate", headers=auth)
    assert dup.status_code == 201 and dup.json()["title"].endswith("(copy)")

    assert client.delete(f"/api/projects/{p['id']}", headers=auth).status_code == 204
    assert client.get(f"/api/projects/{p['id']}", headers=auth).status_code == 404


def test_ownership_isolated(client, auth, other_auth):
    pid = client.post("/api/projects", json={"title": "Private"}, headers=auth).json()["id"]
    assert client.get(f"/api/projects/{pid}", headers=other_auth).status_code == 404
    assert client.put(f"/api/projects/{pid}", json={"title": "x"}, headers=other_auth).status_code == 404
    assert client.delete(f"/api/projects/{pid}", headers=other_auth).status_code == 404
    assert client.post(f"/api/projects/{pid}/render", headers=other_auth).status_code == 404
    assert client.get("/api/projects", headers=other_auth).json() == []


def test_validation(client, auth):
    pid = client.post("/api/projects", json={}, headers=auth).json()["id"]
    url = f"/api/projects/{pid}"
    assert client.put(url, json={"aspect_ratio": "4:3"}, headers=auth).status_code == 422
    assert client.put(url, json={"speed": 5}, headers=auth).status_code == 422
    assert client.put(url, json={"title": "   "}, headers=auth).status_code == 422
    assert client.put(url, json={"voice": "nope-NopeNeural"}, headers=auth).status_code == 422
    assert client.put(url, json={"avatar_id": "does-not-exist"}, headers=auth).status_code == 422
    assert client.put(url, json={"scenes": [{"text": "x" * 1500}]}, headers=auth).status_code == 422
    assert client.put(url, json={"scenes": [{"text": "hi"}] * 25}, headers=auth).status_code == 422
    assert client.put(url, json={"scenes": [{"text": "hi", "media_asset_id": "0" * 32}]}, headers=auth).status_code == 422
    assert client.post("/api/projects", json={"template_id": "bogus"}, headers=auth).status_code == 422


def test_upload_validation_and_use(client, auth, other_auth):
    buf = io.BytesIO()
    Image.new("RGB", (200, 100), "red").save(buf, "PNG")
    r = client.post("/api/media", headers=auth, data={"kind": "avatar"}, files={"file": ("me.png", buf.getvalue(), "image/png")})
    assert r.status_code == 201, r.text
    asset = r.json()
    assert client.get(asset["url"]).status_code == 200

    bad = client.post("/api/media", headers=auth, data={"kind": "image"}, files={"file": ("x.png", b"not an image", "image/png")})
    assert bad.status_code == 400

    pid = client.post("/api/projects", json={}, headers=auth).json()["id"]
    ok = client.put(f"/api/projects/{pid}", json={"avatar_id": f"asset:{asset['id']}"}, headers=auth)
    assert ok.status_code == 200 and ok.json()["avatar_id"] == f"asset:{asset['id']}"
    pid2 = client.post("/api/projects", json={}, headers=other_auth).json()["id"]
    assert client.put(f"/api/projects/{pid2}", json={"avatar_id": f"asset:{asset['id']}"}, headers=other_auth).status_code == 422


def test_render_requires_script_and_daily_cap(client, auth):
    pid = client.post("/api/projects", json={}, headers=auth).json()["id"]
    assert client.post(f"/api/projects/{pid}/render", headers=auth).status_code == 422  # empty script
    client.put(f"/api/projects/{pid}", json={"scenes": [{"text": "Hello"}]}, headers=auth)
    from app.db import SessionLocal
    from app.models import RenderJob

    ids = []
    for _ in range(3):  # limit is 3/day in tests
        r = client.post(f"/api/projects/{pid}/render", headers=auth)
        assert r.status_code == 202, r.text
        assert client.post(f"/api/projects/{pid}/render", headers=auth).status_code in (409, 429)
        with SessionLocal() as db:  # mark finished so the next one is allowed
            j = db.get(RenderJob, r.json()["id"])
            j.status = "done"
            j.project.status = "done"
            db.commit()
    r = client.post(f"/api/projects/{pid}/render", headers=auth)
    assert r.status_code == 429 and "limit" in r.json()["detail"].lower()
    assert client.get("/api/usage", headers=auth).json() == {"used_today": 3, "limit": 3}


def test_import_text(client, auth):
    text = ("Video is the most engaging medium on the internet today. Companies use video to explain products. "
            "Creating video used to require cameras and studios. Now AI avatars can present your script for you. "
            "This makes video creation accessible to everyone. Anyone can type a script and get a video.")
    r = client.post("/api/import/text", json={"text": text, "max_scenes": 3}, headers=auth)
    assert r.status_code == 200 and 1 <= len(r.json()["scenes"]) <= 3
    assert client.post("/api/import/url", json={"url": "http://127.0.0.1:8000/"}, headers=auth).status_code == 400


def test_sceneries_and_intro_settings(client, auth):
    sc = client.get("/api/sceneries").json()
    assert len(sc) >= 6 and client.get(sc[0]["url"]).status_code == 200
    pid = client.post("/api/projects", json={}, headers=auth).json()["id"]
    url = f"/api/projects/{pid}"
    r = client.put(url, headers=auth, json={
        "intro_enabled": False, "intro_title": "Hello",
        "background": {"type": "scenery", "scenery_id": sc[0]["id"]},
        "scenes": [{"text": "a", "background": {"type": "scenery", "scenery_id": sc[1]["id"]}}],
    })
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["intro_enabled"] is False and out["intro_title"] == "Hello" and out["gestures_enabled"] is True
    assert out["scenes"][0]["background"]["scenery_id"] == sc[1]["id"]
    assert client.put(url, headers=auth, json={"background": {"type": "scenery", "scenery_id": "nope"}}).status_code == 422


def test_avatar_upload_reports_meta(client, auth):
    import numpy as np

    buf = io.BytesIO()
    Image.fromarray((np.random.default_rng(1).random((240, 200, 3)) * 255).astype("uint8")).save(buf, "PNG")
    r = client.post("/api/media", headers=auth, data={"kind": "avatar", "gender": "male", "remove_bg": "true"}, files={"file": ("p.png", buf.getvalue(), "image/png")})
    assert r.status_code == 201, r.text
    a = r.json()
    assert a["gender"] == "male" and a["face_found"] is False and (a["width"], a["height"]) == (768, 768)
    assert client.post("/api/media", headers=auth, data={"kind": "avatar", "gender": "robot"}, files={"file": ("p.png", buf.getvalue(), "image/png")}).status_code == 422


def test_fine_tune_mouth(client, auth, other_auth):
    import numpy as np

    buf = io.BytesIO()
    Image.fromarray((np.random.default_rng(2).random((200, 160, 3)) * 255).astype("uint8")).save(buf, "PNG")
    a = client.post("/api/media", headers=auth, data={"kind": "avatar"}, files={"file": ("p.png", buf.getvalue(), "image/png")}).json()
    r = client.put(f"/api/media/{a['id']}/mouth", headers=auth, json={"x": 0.5, "y": 0.62, "width": 0.2})
    assert r.status_code == 200 and r.json()["mouth"] == [0.5, 0.62, 0.2]
    assert client.get("/api/media?kind=avatar", headers=auth).json()[0]["mouth"] == [0.5, 0.62, 0.2]
    assert client.put(f"/api/media/{a['id']}/mouth", headers=auth, json={"x": 0.5, "y": 2, "width": 0.2}).status_code == 422
    assert client.put(f"/api/media/{a['id']}/mouth", headers=other_auth, json={"x": 0.5, "y": 0.6, "width": 0.2}).status_code == 404
