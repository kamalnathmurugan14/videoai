"""SadTalker integration (https://github.com/OpenTalker/SadTalker, Apache-2.0): neural, audio-driven talking head from a
single photo — real lip-sync, expressions, blinks and head motion. It runs in its own Python 3.10 environment (installed by
setup-sadtalker.ps1) and is called as a subprocess, so its old pinned dependencies never clash with this app's."""
import shutil
import subprocess
import time
from pathlib import Path

from ..config import get_settings

CHECKPOINTS = ("mapping_00109-model.pth.tar", "mapping_00229-model.pth.tar")
ENHANCER_WEIGHTS = ("GFPGANv1.4.pth", "detection_Resnet50_Final.pth", "parsing_parsenet.pth", "alignment_WFLW_4HG.pth")


class SadTalkerError(RuntimeError):
    pass


def _paths():
    s = get_settings()
    d = Path(s.sadtalker_dir).expanduser() if s.sadtalker_dir else None
    py = Path(s.sadtalker_python).expanduser() if s.sadtalker_python else None
    return s, d, py


def status() -> dict:
    """Is SadTalker installed and complete? Lists every missing piece so the System check can say what to fix."""
    s, d, py = _paths()
    problems = []
    if not d:
        return {"configured": False, "ready": False, "problems": ["SADTALKER_DIR is not set (run setup-sadtalker.ps1)"]}
    if not (d / "inference.py").exists():
        problems.append(f"inference.py not found in {d}")
    if not py or not py.exists():
        problems.append(f"SadTalker Python not found: {py or '(SADTALKER_PYTHON not set)'}")
    ck = d / "checkpoints"
    for f in CHECKPOINTS:
        if not (ck / f).exists():
            problems.append(f"missing checkpoint {f}")
    if not (ck / f"SadTalker_V0.0.2_{s.sadtalker_size}.safetensors").exists():
        problems.append(f"missing checkpoint SadTalker_V0.0.2_{s.sadtalker_size}.safetensors")
    if s.sadtalker_enhancer == "gfpgan":
        for f in ENHANCER_WEIGHTS:
            if not (d / "gfpgan" / "weights" / f).exists():
                problems.append(f"missing enhancer weight gfpgan/weights/{f}")
    return {"configured": True, "ready": not problems, "problems": problems, "dir": str(d), "enhancer": s.sadtalker_enhancer or None,
            "size": s.sadtalker_size}


def run(image: Path, audio: Path, out_dir: Path) -> Path:
    """Run SadTalker on one photo + WAV; returns the generated MP4. Raises SadTalkerError with the tool's own message."""
    s, d, py = _paths()
    st = status()
    if not st["ready"]:
        raise SadTalkerError("SadTalker is not installed: " + "; ".join(st["problems"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(py), "inference.py", "--driven_audio", str(audio.resolve()), "--source_image", str(image.resolve()),
           "--result_dir", str(out_dir.resolve()), "--checkpoint_dir", str((d / "checkpoints").resolve()),
           "--still", "--preprocess", "full", "--size", str(s.sadtalker_size)]
    if s.sadtalker_enhancer:
        cmd += ["--enhancer", s.sadtalker_enhancer]
    t = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(d), capture_output=True, text=True, timeout=s.sadtalker_timeout_seconds)
    except subprocess.TimeoutExpired:
        raise SadTalkerError(f"SadTalker took longer than {s.sadtalker_timeout_seconds}s (very slow without an NVIDIA GPU)")
    # inference.py moves its final result to <result_dir>/<timestamp>.mp4; intermediates live in sub-folders
    videos = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime) or sorted(out_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if r.returncode != 0 or not videos:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-6:]
        raise SadTalkerError("SadTalker failed: " + " | ".join(tail)[-600:])
    result = videos[-1]
    (out_dir / "_timing.txt").write_text(f"{time.time() - t:.1f}")
    return result


def device_info() -> dict:
    """Which device SadTalker's PyTorch will use (cuda = NVIDIA GPU, fast; cpu = slow)."""
    _, _, py = _paths()
    if not py or not py.exists():
        return {"device": None, "error": "SadTalker Python not found"}
    code = "import torch;print('cuda' if torch.cuda.is_available() else 'cpu');print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    try:
        r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=120)
        lines = r.stdout.strip().splitlines()
        if r.returncode != 0 or not lines:
            return {"device": None, "error": (r.stderr or "")[-300:]}
        return {"device": lines[0], "gpu": lines[1] if len(lines) > 1 else ""}
    except Exception as e:
        return {"device": None, "error": str(e)[:300]}
