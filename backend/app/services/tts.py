"""Text-to-speech with a provider fallback chain: edge-tts -> gTTS -> espeak-ng (-> synthetic, tests only)."""
import asyncio
import logging
import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

from ..config import get_settings
from .catalog import VOICES

log = logging.getLogger(__name__)
SAMPLE_RATE = 24000


class TTSError(RuntimeError):
    pass


def _lang(voice: str) -> tuple[str, str]:
    """'en-GB-RyanNeural' -> ('en', 'en-GB')"""
    parts = voice.split("-")
    full = "-".join(parts[:2]) if len(parts) >= 2 else parts[0]
    return parts[0].lower(), full


_GENDER = {v[0]: v[3] for v in VOICES}


def voice_gender(voice: str) -> str:
    return _GENDER.get(voice, "female")


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

    try:
        asyncio.run(run())
    except Exception:  # the Microsoft service occasionally drops a connection: one retry before falling back
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
    male = voice_gender(voice) == "male"
    v = f"{full.lower() if lang == 'en' else lang}{'+m3' if male else '+f3'}"
    r = subprocess.run([exe, "-v", v, "-s", str(int(165 * speed)), "-w", str(raw), text], capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or not raw.exists():
        raise TTSError(f"espeak failed: {r.stderr[-200:]}")
    _to_wav(raw, out)
    raw.unlink(missing_ok=True)


_PYTTSX3_SCRIPT = r"""
import sys, pyttsx3
text, out, gender, lang, rate = sys.argv[1:6]
e = pyttsx3.init()
voices = e.getProperty("voices")
def score(v):
    blob = (v.name + " " + str(getattr(v, "gender", "")) + " " + " ".join(str(l) for l in (v.languages or [])) + " " + v.id).lower()
    s = 0
    male = ("david", "mark", "george", "ravi", "male", "alex", "daniel", "fred")
    female = ("zira", "hazel", "susan", "heera", "female", "samantha", "victoria", "karen")
    if gender == "male" and any(k in blob for k in male) and "female" not in blob: s += 2
    if gender == "female" and any(k in blob for k in female): s += 2
    if lang in blob: s += 1
    return s
if voices:
    e.setProperty("voice", max(voices, key=score).id)
e.setProperty("rate", int(float(rate)))
e.save_to_file(text, out)
e.runAndWait()
"""


def _pyttsx3(text: str, voice: str, speed: float, out: Path) -> None:
    lang, _ = _lang(voice)
    raw = out.with_suffix(".sapi.wav")
    r = subprocess.run([sys.executable, "-c", _PYTTSX3_SCRIPT, text, str(raw), voice_gender(voice), lang, str(175 * speed)],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or not raw.exists() or raw.stat().st_size < 1000:
        raise TTSError(f"pyttsx3 failed: {r.stderr[-200:]}")
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


ENGINES = {"edge": _edge, "pyttsx3": _pyttsx3, "gtts": _gtts, "espeak": _espeak, "synthetic": _synthetic}


def synthesize(text: str, voice: str, speed: float, out: Path) -> tuple[str, list[str]]:
    """Write mono 24 kHz WAV to `out`. Returns (engine that worked, errors from engines tried before it)."""
    errors = []
    for name in get_settings().tts_list:
        fn = ENGINES.get(name)
        if not fn:
            continue
        try:
            fn(text, voice, speed, out)
            if out.exists() and out.stat().st_size > 1000:
                return name, errors
        except Exception as e:  # network down, missing engine, bad voice ...
            errors.append(f"{name}: {str(e)[:140]}")
            log.warning("TTS engine %s failed: %s", name, e)
    raise TTSError("No text-to-speech engine worked (" + "; ".join(errors) + ")")


def wav_samples(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
    return data, sr
