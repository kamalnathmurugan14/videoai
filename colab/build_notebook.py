"""Builds VideoAI_GPU_worker.ipynb from gpu_worker.py (run: python colab/build_notebook.py)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
worker = (HERE / "gpu_worker.py").read_text()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text, title=None):
    meta = {"cellView": "form"} if title else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(True)}


SAD_PINS = ("numpy==1.23.4 face_alignment==1.3.5 imageio==2.19.3 imageio-ffmpeg==0.4.7 librosa==0.9.2 numba==0.58.1 "
            "resampy==0.3.1 pydub==0.25.1 scipy==1.10.1 kornia==0.6.8 tqdm yacs==0.1.8 pyyaml joblib==1.1.0 "
            "scikit-image==0.19.3 opencv-python==4.7.0.72 basicsr==1.4.2 facexlib==0.3.0 gfpgan==1.3.8 av safetensors")

cells = [
    md("""
# VideoAI GPU worker (free Google Colab GPU)

This notebook gives the **VideoAI** app a free NVIDIA GPU so it can generate realistic talking-avatar videos:

* **EchoMimicV2** – half body with **hand gestures** from one photo + voice (closest to HeyGen). Apache-2.0.
* **SadTalker** – realistic face / lip-sync. Apache-2.0. Used automatically if EchoMimicV2 fails (e.g. out of GPU memory).

**How to use**
1. Menu **Runtime → Change runtime type → T4 GPU → Save**.
2. Menu **Runtime → Run all** (first run downloads ~15 GB and takes roughly 20–30 minutes).
3. The last cell prints a **URL** and a **TOKEN**. In VideoAI open **System check → GPU worker**, paste both, press *Connect*.
4. In the editor choose **Look & feel → Animation engine → HeyGen-like (EchoMimicV2, Colab GPU)** and render.
5. **Keep this tab open** while rendering. Free Colab disconnects after inactivity / a few hours; then just *Run all* again and paste the new URL.

Rendering on the free T4 GPU is slow (EchoMimicV2: very roughly 10–30 minutes per 10 seconds of video; SadTalker ~1 minute).
"""),
    code("""
#@title 1. Check the GPU
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "NO GPU: Runtime > Change runtime type > T4 GPU"
""", True),
    code("""
#@title 2. Options
INSTALL_ECHOMIMIC = True  #@param {type:"boolean"}
INSTALL_SADTALKER = True  #@param {type:"boolean"}
""", True),
    code("""
#@title 3. Base tools (ffmpeg, uv, Python 3.10 for the models)
!apt-get -qq update > /dev/null && apt-get -qq install -y ffmpeg git-lfs > /dev/null
!pip -q install uv fastapi "uvicorn[standard]" python-multipart huggingface_hub
!uv python install 3.10
""", True),
    code(f"""
#@title 4. Install SadTalker (face / lip-sync)
import subprocess
if INSTALL_SADTALKER:
    script = r'''
set -e
cd /content
[ -d SadTalker ] || git clone --depth 1 https://github.com/OpenTalker/SadTalker.git
[ -x env_sadtalker/bin/python ] || uv venv --python 3.10 env_sadtalker
P=/content/env_sadtalker/bin/python
uv pip install --python $P "setuptools<70" wheel
uv pip install --python $P torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
uv pip install --python $P {SAD_PINS}
cd SadTalker
mkdir -p checkpoints gfpgan/weights
R=https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc
for f in mapping_00109-model.pth.tar mapping_00229-model.pth.tar SadTalker_V0.0.2_256.safetensors SadTalker_V0.0.2_512.safetensors; do
  [ -s checkpoints/$f ] || wget -q -O checkpoints/$f $R/$f
done
[ -s gfpgan/weights/alignment_WFLW_4HG.pth ] || wget -q -O gfpgan/weights/alignment_WFLW_4HG.pth https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth
[ -s gfpgan/weights/detection_Resnet50_Final.pth ] || wget -q -O gfpgan/weights/detection_Resnet50_Final.pth https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth
[ -s gfpgan/weights/GFPGANv1.4.pth ] || wget -q -O gfpgan/weights/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
[ -s gfpgan/weights/parsing_parsenet.pth ] || wget -q -O gfpgan/weights/parsing_parsenet.pth https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth
echo SADTALKER_OK
'''
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    print("SadTalker installed" if "SADTALKER_OK" in r.stdout else "SadTalker install FAILED (see the messages above)")
""", True),
    code("""
#@title 5. Install EchoMimicV2 (half body + hand gestures)
import subprocess
if INSTALL_ECHOMIMIC:
    script = r'''
set -e
cd /content
[ -d echomimic_v2 ] || git clone --depth 1 https://github.com/antgroup/echomimic_v2.git
[ -x env_echomimic/bin/python ] || uv venv --python 3.10 env_echomimic
P=/content/env_echomimic/bin/python
uv pip install --python $P torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python $P -r echomimic_v2/requirements.txt
uv pip install --python $P --no-deps facenet_pytorch==2.6.0
cd echomimic_v2
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("BadToBest/EchoMimicV2", local_dir="pretrained_weights")
snapshot_download("stabilityai/sd-vae-ft-mse", local_dir="pretrained_weights/sd-vae-ft-mse")
snapshot_download("lambdalabs/sd-image-variations-diffusers", local_dir="pretrained_weights/sd-image-variations-diffusers",
                  allow_patterns=["unet/*", "image_encoder/*", "feature_extractor/*", "scheduler/*", "model_index.json"])
PY
mkdir -p pretrained_weights/audio_processor
[ -s pretrained_weights/audio_processor/tiny.pt ] || wget -q -O pretrained_weights/audio_processor/tiny.pt https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt
echo ECHOMIMIC_OK
'''
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    print("EchoMimicV2 installed" if "ECHOMIMIC_OK" in r.stdout else "EchoMimicV2 install FAILED - the worker will use SadTalker instead")
""", True),
    code("%%writefile /content/gpu_worker.py\n" + worker, None),
    code("""
#@title 6. Self-test both engines on their own example files (shows real speed on this GPU)
import glob, os, subprocess, sys, time, wave
sys.path.insert(0, "/content")
os.environ["WORKER_STUB"] = "0"
import importlib, gpu_worker
importlib.reload(gpu_worker)
from pathlib import Path

def short_wav(src, dst, seconds=3):
    r = wave.open(src); w = wave.open(dst, "wb"); w.setparams(r.getparams())
    w.writeframes(r.readframes(int(r.getframerate() * seconds))); w.close()

job = Path("/content/selftest"); job.mkdir(exist_ok=True)
for name in ("sadtalker", "echomimic"):
    st = gpu_worker.engine_status(name)
    if not st["ready"]:
        print(f"{name}: not installed {st['problems']}"); continue
    d = gpu_worker.ENG[name][0]
    imgs = sorted(glob.glob(str(d / ("examples/source_image/*.png" if name == "sadtalker" else "assets/halfbody_demo/refimag/**/*.png")), recursive=True))
    wavs = sorted(glob.glob(str(d / ("examples/driven_audio/*.wav" if name == "sadtalker" else "assets/halfbody_demo/audio/**/*.wav")), recursive=True))
    if not imgs or not wavs:
        print(f"{name}: no example files found"); continue
    short_wav(wavs[0], str(job / f"{name}.wav"))
    t = time.time()
    try:
        if name == "sadtalker":
            out = gpu_worker.run_sadtalker(Path(imgs[0]), job / f"{name}.wav", job)
        else:
            out, _ = gpu_worker.run_echomimic(Path(imgs[0]), job / f"{name}.wav", job, 24)
        print(f"{name}: OK, 3 s of video in {time.time() - t:.0f} s -> {out}")
    except Exception as e:
        print(f"{name}: FAILED after {time.time() - t:.0f} s: {e}")
""", True),
    code("""
#@title 7. Start the worker and get the URL + TOKEN for VideoAI (keep this tab open)
import os, re, secrets, subprocess, threading, time
TOKEN = secrets.token_urlsafe(12)
env = {**os.environ, "WORKER_TOKEN": TOKEN, "WORKER_STUB": "0"}
server = subprocess.Popen(["python", "-m", "uvicorn", "gpu_worker:app", "--host", "127.0.0.1", "--port", "8765"], cwd="/content", env=env)
if not os.path.exists("/content/cloudflared"):
    subprocess.run(["wget", "-q", "-O", "/content/cloudflared",
                    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"], check=True)
    os.chmod("/content/cloudflared", 0o755)
tunnel = subprocess.Popen(["/content/cloudflared", "tunnel", "--url", "http://127.0.0.1:8765", "--no-autoupdate"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
URL = None
for line in tunnel.stdout:
    m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", line)
    if m:
        URL = m.group(0)
        break
threading.Thread(target=lambda: [None for _ in tunnel.stdout], daemon=True).start()
time.sleep(3)
print("=" * 70)
print("Paste these into VideoAI -> System check -> GPU worker:")
print("URL:  ", URL)
print("TOKEN:", TOKEN)
print("=" * 70)
""", True),
    code("""
#@title 8. (Optional) keep the session busy while rendering - stop this cell when you are done
import time, json, urllib.request
while True:
    try:
        req = urllib.request.Request("http://127.0.0.1:8765/health", headers={"X-Token": TOKEN})
        h = json.loads(urllib.request.urlopen(req, timeout=10).read())
        print(time.strftime("%H:%M:%S"), "worker OK, jobs waiting:", h["queued"])
    except Exception as e:
        print(time.strftime("%H:%M:%S"), "worker not responding:", e)
    time.sleep(60)
""", True),
]

for i, c in enumerate(cells):
    c["id"] = f"cell-{i}"
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}}
(HERE / "VideoAI_GPU_worker.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote", HERE / "VideoAI_GPU_worker.ipynb", len(cells), "cells")
