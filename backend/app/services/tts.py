"""Text-to-speech with a provider fallback chain: edge-tts -> gTTS -> espeak-ng (-> synthetic, tests only)."""
import asyncio
import logging
import math
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from ..config import get_settings

log = logging.getLogger(__name__)
SAMPLE_RATE = 24000


class TTSError(RuntimeError):
    pass


def _lang(voice: str) -> tuple[str, str]:
    """'en-GB-RyanNeural' -> ('en', 'en-GB')"""
    parts = voice.split("-")
    full = "-".join(parts[:2]) if len(parts) >= 2 else parts[0]
    return parts[0].lower(), full


def _ffmpeg(args: list[str]) -> None:
    r = subprocess.run([get_settings().ffmpeg_bin, "-y", "-loglevel", "error", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise TTSError(f"ffmpeg failed: {r.stderr[-300:]}")


def _to_wav(src: Path, dst: Path, tempo: float = 1.0) -> None:
    af = []
    if abs(tempo - 1.0) > 0.02:
        t = min(2.0, max(0.5, tempo))
        af = ["-af", f"atempo={t:.3f}"]
    _ffmpeg(["-i", str(src), *af, "-ar", str(SAMPLE_RATE), "-ac", "1", str(dst)])


def _edge(text: str, voice: str, speed: float, out: Path) -> None:
    import edge_tts

    rate = f"{round((speed - 1) * 100):+d}%"
    mp3 = out.with_suffix(".edge.mp3")

    async def run():
        await asyncio.wait_for(edge_tts.Communicate(text, voice, rate=rate).save(str(mp3)), get_settings().tts_timeout_seconds)

    asyncio.run(run())
    if not mp3.exists() or mp3.stat().st_size < 500:
        raise TTSError("edge-tts returned no audio")
    _to_wav(mp3, out)
    mp3.unlink(missing_ok=True)


def _gtts(text: str, voice: str, speed: float, out: Path) -> None:
    from gtts import gTTS

    lang, full = _lang(voice)
    tld = {"en-GB": "co.uk", "en-AU": "com.au", "en-IN": "co.in", "es-MX": "com.mx", "pt-BR": "com.br", "fr-FR": "fr"}.get(full, "com")
    lang = {"zh": "zh-CN"}.get(lang, lang)
    mp3 = out.with_suffix(".gtts.mp3")
    gTTS(text=text, lang=lang, tld=tld, timeout=get_settings().tts_timeout_seconds).save(str(mp3))
    _to_wav(mp3, out, tempo=speed)  # gTTS has no rate control
    mp3.unlink(missing_ok=True)


def _espeak(text: str, voice: str, speed: float, out: Path) -> None:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        raise TTSError("espeak-ng not installed")
    lang, full = _lang(voice)
    raw = out.with_suffix(".espeak.wav")
    male = voice.lower().split("-")[-1].startswith(("guy", "ryan", "henri", "conrad", "jorge", "prabhat", "madhur", "valluvar"))
    v = f"{full.lower() if lang == 'en' else lang}{'+m3' if male else '+f3'}"
    r = subprocess.run([exe, "-v", v, "-s", str(int(165 * speed)), "-w", str(raw), text], capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or not raw.exists():
        raise TTSError(f"espeak failed: {r.stderr[-200:]}")
    _to_wav(raw, out)
    raw.unlink(missing_ok=True)


def _synthetic(text: str, voice: str, speed: float, out: Path) -> None:
    """Deterministic speech-like babble (syllable-modulated tones). For tests / no-engine environments only."""
    words = max(1, len(text.split()))
    dur = max(0.8, words * 0.35 / speed)
    t = np.arange(int(dur * SAMPLE_RATE)) / SAMPLE_RATE
    f0 = 120 + (hash(voice) % 80)
    syl = np.abs(np.sin(math.pi * 3.8 * speed * t)) ** 1.5
    gaps = (np.sin(2 * math.pi * 0.6 * t) > -0.7).astype(float)
    sig = sum(np.sin(2 * math.pi * f0 * k * t) / k for k in (1, 2, 3, 4)) * syl * gaps * 0.2
    pcm = (np.clip(sig, -1, 1) * 32767).astype("<i2")
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


ENGINES = {"edge": _edge, "gtts": _gtts, "espeak": _espeak, "synthetic": _synthetic}


def synthesize(text: str, voice: str, speed: float, out: Path) -> str:
    """Write mono 24 kHz WAV to `out`; returns name of the engine that worked."""
    errors = []
    for name in get_settings().tts_list:
        fn = ENGINES.get(name)
        if not fn:
            continue
        try:
            fn(text, voice, speed, out)
            if out.exists() and out.stat().st_size > 1000:
                return name
        except Exception as e:  # network down, missing engine, bad voice ...
            errors.append(f"{name}: {str(e)[:120]}")
            log.warning("TTS engine %s failed: %s", name, e)
    raise TTSError("No text-to-speech engine worked (" + "; ".join(errors) + ")")


def wav_samples(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
    return data, sr
