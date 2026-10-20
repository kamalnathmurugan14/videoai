"""Person segmentation with Google's MediaPipe "selfie multiclass" model (Apache-2.0 TFLite in assets/models), run via
ai-edge-litert. Classes: background, hair, body skin, face skin, clothes, others; we use 1 - P(background). Unlike
colour-based GrabCut this keeps clothes/shoulders that look similar to the background and drops busy scenery."""
import logging
import threading
from functools import lru_cache

import cv2
import numpy as np

from ..config import ASSETS_DIR

log = logging.getLogger(__name__)
_lock = threading.Lock()


@lru_cache(maxsize=1)
def _interpreter():
    try:
        from ai_edge_litert.interpreter import Interpreter

        it = Interpreter(model_path=str(ASSETS_DIR / "models" / "selfie_multiclass_256x256.tflite"))
        it.allocate_tensors()
        return it
    except Exception as e:
        log.warning("person segmentation unavailable: %s", e)
        return None


def available() -> bool:
    return _interpreter() is not None


def person_probability(rgb: np.ndarray) -> np.ndarray | None:
    """HxW float32 probability that each pixel belongs to a person (any person), or None if the model is unavailable."""
    it = _interpreter()
    if it is None:
        return None
    h, w = rgb.shape[:2]
    s = 256 / max(h, w)  # letterbox so the person is not squashed
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    pad = np.zeros((256, 256, 3), np.float32)
    oy, ox = (256 - nh) // 2, (256 - nw) // 2
    pad[oy:oy + nh, ox:ox + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    with _lock:
        inp = it.get_input_details()[0]
        it.set_tensor(inp["index"], pad[None])
        it.invoke()
        out = it.get_tensor(it.get_output_details()[0]["index"])[0]
    e = np.exp(out - out.max(axis=-1, keepdims=True))
    prob = 1.0 - (e[..., 0] / e.sum(axis=-1))
    prob = prob[oy:oy + nh, ox:ox + nw]
    return cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
