"""Royalty-free background music: tiny procedural loops generated on first use (nothing is committed)."""
import wave

import numpy as np

from . import storage

SR = 22050
TRACKS = {
    "calm": {"name": "Calm pad", "chords": [(57, 60, 64), (53, 57, 60), (55, 59, 62), (52, 55, 59)], "bpm": 60, "arp": False},
    "upbeat": {"name": "Upbeat pulse", "chords": [(60, 64, 67), (65, 69, 72), (62, 65, 69), (67, 71, 74)], "bpm": 112, "arp": True},
    "cinematic": {"name": "Cinematic rise", "chords": [(45, 52, 57), (43, 50, 55), (41, 48, 53), (43, 50, 55)], "bpm": 72, "arp": False},
}


def _synth(cfg: dict) -> np.ndarray:
    beat = 60 / cfg["bpm"]
    bar = beat * 4
    out = []
    for chord in cfg["chords"]:
        t = np.arange(int(bar * SR)) / SR
        sig = np.zeros_like(t)
        for i, note in enumerate(chord):
            f = 440 * 2 ** ((note - 69) / 12)
            sig += np.sin(2 * np.pi * f * t) * 0.18 + np.sin(2 * np.pi * f * 2 * t) * 0.04
            if cfg["arp"]:
                step = (t / (beat / 2)).astype(int) % len(chord) == i
                env = np.exp(-((t % (beat / 2)) * 9))
                sig += np.sin(2 * np.pi * f * 2 * t) * env * step * 0.25
        fade = np.minimum(1, np.minimum(t / 0.15, (bar - t) / 0.15))
        out.append(sig * (fade if not cfg["arp"] else 1) * 0.6)
    audio = np.concatenate(out)
    return (np.clip(audio / max(1e-6, np.abs(audio).max()) * 0.8, -1, 1) * 32767).astype("<i2")


def ensure_library() -> dict[str, str]:
    d = storage.subdir("music")
    for tid, cfg in TRACKS.items():
        p = d / f"{tid}.wav"
        if not p.exists():
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(_synth(cfg).tobytes())
    return {tid: str(d / f"{tid}.wav") for tid in TRACKS}


def listing() -> list[dict]:
    ensure_library()
    return [{"id": k, "name": v["name"], "url": f"/storage/music/{k}.wav"} for k, v in TRACKS.items()]


def path_for(track_id: str | None):
    if track_id in TRACKS:
        return storage.subdir("music") / f"{track_id}.wav"
    return None
