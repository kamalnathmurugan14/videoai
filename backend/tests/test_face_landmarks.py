"""Regression tests for photo -> avatar landmark placement (uses a public-domain NASA portrait from scikit-image)."""
import numpy as np
import pytest
from PIL import Image

from app.services import face, landmarks


def test_anchor_layout():
    a = landmarks._anchors()
    assert a.shape == (896, 2) and a.min() > 0 and a.max() < 1


def test_falls_back_to_haar_when_models_unavailable(monkeypatch):
    monkeypatch.setattr(landmarks, "_interpreters", lambda: None)
    rgb = (np.random.default_rng(0).random((200, 160, 3)) * 255).astype("uint8")
    info = face.analyze(rgb)
    assert info["source"] in ("haar", "guess") and info["found"] is False  # noise has no face, but nothing crashes
    canvas, meta = face.build_avatar(Image.fromarray(rgb), True, "female")
    assert canvas.size == (face.CANVAS, face.CANVAS) and "No face was detected" in meta["warning"]


def test_cutout_rejects_implausible_masks():
    rgb = (np.random.default_rng(1).random((300, 300, 3)) * 255).astype("uint8")
    info = {"face_w": 100.0, "chin": 200.0, "eyes": [(120.0, 120.0, 10.0), (180.0, 120.0, 10.0)]}
    out = face.cutout_alpha(rgb, info)
    assert out is None or out.dtype == np.uint8  # noise has no person: either rejected or a valid mask, never a crash


@pytest.fixture(scope="module")
def astronaut():
    data = pytest.importorskip("skimage.data")
    return Image.fromarray(data.astronaut())


def test_mouth_lands_on_the_lips(astronaut):
    rgb = np.asarray(astronaut)
    lm = landmarks.detect_landmarks(rgb)
    assert lm is not None
    sm = landmarks.summarize(lm)
    mx, my, mw = sm["mouth"]
    # reference positions measured by eye on the 512x512 portrait: lips at x~223, y~146 (inner lip line), width ~43
    assert abs(mx - 223) < 8 and abs(my - 146) < 7 and 30 < mw < 60
    assert sm["already_open"] is True  # she is smiling with teeth showing
    left, right = sm["eyes"]
    assert left[0] < mx < right[0] and left[1] < my and abs(left[1] - right[1]) < 8


def test_avatar_mouth_is_consistent_after_rotation_and_flip(astronaut):
    base_canvas, base_meta = face.build_avatar(astronaut, True, "female")
    assert base_meta["source"] == "landmarks" and base_meta["found"] is True
    for variant in (astronaut.rotate(20, resample=Image.BICUBIC, fillcolor=(120, 120, 120)),
                    astronaut.rotate(-25, resample=Image.BICUBIC, fillcolor=(120, 120, 120)),
                    astronaut.transpose(Image.FLIP_LEFT_RIGHT)):
        canvas, meta = face.build_avatar(variant, True, "female")
        assert meta["source"] == "landmarks"
        # the tilted/flipped photo is normalised so the mouth ends up in (nearly) the same canvas position
        assert abs(meta["mouth"][0] - base_meta["mouth"][0]) < 0.03 and abs(meta["mouth"][1] - base_meta["mouth"][1]) < 0.03
        assert abs(meta["eyes"][0][1] - meta["eyes"][1][1]) < 0.02  # eyes level
        assert canvas.getchannel("A").getextrema()[1] > 200  # not a blank avatar


def test_background_removal_keeps_the_whole_head(astronaut):
    canvas, meta = face.build_avatar(astronaut, True, "female")
    alpha = np.asarray(canvas.getchannel("A")).astype(float) / 255
    ex = meta["eyes"][0][0] * face.CANVAS, meta["eyes"][1][0] * face.CANVAS
    top_rows = alpha[int(0.02 * face.CANVAS): int(0.2 * face.CANVAS), int(ex[0]): int(ex[1])]
    assert meta["cutout"] and top_rows.mean() > 0.5  # hair / top of the head is opaque, not clipped flat
