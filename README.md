# VideoAI — free AI avatar video creator

A free, self-hostable alternative to HeyGen / Vidnoz. Write a multi-scene script, pick an animated avatar and a voice,
and render a captioned MP4 — no paid APIs required.

**Stack:** React 19 + Vite + React Router + Tailwind · FastAPI (Python 3.11+) · SQLAlchemy 2 + Alembic ·
SQLite (default) or PostgreSQL · FFmpeg · edge-tts / gTTS / espeak-ng · Pillow + NumPy.

## Features

- **Auth:** email/password (bcrypt + JWT) and one-click **guest mode**.
- **Dashboard:** list, create (blank or from template), rename, duplicate, delete, status
  (`draft → queued → rendering → done / failed`), preview player, MP4 download.
- **Editor:** 6 bundled avatars + upload your own photo (face detection, optional background removal) · multi-scene
  script (per-scene text, background, image) · 8 scenery backgrounds (office, beach, city, forest, ...) assignable per
  scene · male/female voice filter, 24 voices, speed, voice preview that tells you which engine really played ·
  colour / gradient / image backgrounds · intro title card ·
  auto-captions (3 styles, word highlight) · 16:9 / 9:16 / 1:1 · templates (explainer, social promo, product intro) ·
  background music (3 procedural royalty-free loops) · text / blog-URL → scenes.
- **Render pipeline:** DB-backed job queue + background worker → per-scene TTS → animated 2D "puppet" avatar (jaw/mouth
  driven by the audio, plus breathing, body sway, shoulder shrugs, head roll/nod/turn, eyebrows, blinks and illustrated
  hand gestures that follow the speech) → captions → slowly panning
  scenery, intro title + avatar entrance, crossfades between scenes → FFmpeg encode → concat (+ music) → MP4 + thumbnail.
  Progress via polling.
- **Limits:** per-user and per-IP daily render caps, upload size/type validation (images are re-encoded), scene/length caps.

## Run locally (without Docker)

Prerequisites: Python 3.11+, Node 20+, **FFmpeg** on `PATH`. Optional: `espeak-ng` (offline voice fallback).

```bash
# 1) Backend  (terminal 1)
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example .env                                # edit SECRET_KEY at least
uvicorn app.main:app --reload --port 8000
#   Migrations run automatically on startup. API docs: http://localhost:8000/docs

# 2) Frontend (terminal 2)
cd frontend
npm install
npm run dev                                            # http://localhost:5173  (proxies /api and /storage to :8000)
```

Open http://localhost:5173, click **Try it now**, create a video from a template, and press **Render video**.

Install FFmpeg: `sudo apt install ffmpeg espeak-ng` · `brew install ffmpeg espeak-ng` · `choco install ffmpeg`.

### Tests

```bash
cd backend && pip install -r requirements-dev.txt && pytest
```
Covers auth, project CRUD/ownership/validation, uploads, daily cap, caption timing, and the render pipeline end to end
(API → queue → worker → real H.264/AAC MP4). Tests use a built-in synthetic voice, so they need FFmpeg but no network.

## Updating

```powershell
# Windows (PowerShell), from the project folder, with your virtualenv activated:
.\update-windows.ps1
```
```bash
# macOS / Linux
git fetch origin main && git reset --hard origin/main
(cd backend && pip install -r requirements.txt) && (cd frontend && npm install)
```
Then restart the backend and frontend. Open **http://localhost:5173/system** (or the *System check* link in the footer)
to confirm the version, that precise face tracking / the person cut-out model are active, and to test which voice
engines work on your computer. `git reset --hard` discards local edits to tracked files (typically a `package-lock.json`
changed by `npm install`, which otherwise makes `git pull` refuse to update); your `.env`, database and media are kept.

## Run with Docker Compose

```bash
cp .env.example .env            # set SECRET_KEY and POSTGRES_PASSWORD
docker compose up --build
```
App: http://localhost:8080 (nginx serves the frontend and proxies `/api`, `/storage` to the backend).
Services: `db` (PostgreSQL 16), `backend` (FastAPI + FFmpeg + espeak-ng + worker thread), `frontend` (nginx).
Data lives in the `pgdata` and `storage` volumes.

## Configuration

All settings are environment variables — see [`.env.example`](.env.example). Highlights:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | `sqlite:///./storage/videoai.db` (default) or `postgresql+psycopg2://user:pw@host/db` |
| `SECRET_KEY` | JWT signing key (the server refuses the default when `ENVIRONMENT=production`) |
| `RENDER_DAILY_LIMIT_USER` / `_IP` | Free-tier render caps per UTC day |
| `RENDER_SCALE` | `1.0` = 1280×720; `0.5` renders ~3–4× faster |
| `TTS_PROVIDERS` | Engine fallback order: `edge,gtts,espeak` |
| `WORKER_ENABLED` | `false` to run the worker separately: `python -m app.workers.run` |
| `AVATAR_PROVIDER` / `AVATAR_COMMAND` | Plug in a real lip-sync model (below) |
| `ANTHROPIC_API_KEY` | Optional: LLM summarisation for text/URL import (heuristic otherwise) |

## Architecture

```
backend/app
  api/        auth, projects (+render/download/usage), media uploads, catalog (avatars/voices/templates/music/import)
  models/     SQLAlchemy 2 models: User, MediaAsset, Project, Scene, RenderJob
  schemas/    pydantic v2 request/response models
  services/   tts, avatar (provider interface), captions, renderer, music, script_import, storage
  workers/    DB-backed queue: claim_next() → process_job(); runs as a thread or standalone
backend/alembic/   migrations (applied on startup)      backend/assets/avatars/   bundled avatar art
frontend/src/      pages (Landing, Auth, Dashboard, Editor, VideoDetail), api client, auth context
```

**Job queue:** `POST /api/projects/{id}/render` inserts a `RenderJob(queued)`. The worker atomically claims the oldest
queued job (`UPDATE … WHERE status='queued'`), writes progress/stage rows as it goes, and on restart re-queues jobs
left `running`. The frontend polls `GET /api/projects/{id}` / `GET /api/jobs/{id}`.

### HeyGen-like videos on a free Google Colab GPU (recommended without an NVIDIA GPU)

HeyGen-style output (real mouth and teeth, head motion, **hand gestures**) needs a neural video model on a GPU. The
notebook [`colab/VideoAI_GPU_worker.ipynb`](colab/VideoAI_GPU_worker.ipynb) runs one on Google's free T4 GPU:

* **EchoMimicV2** (Apache-2.0): half-body animation with hand gestures from one photo + voice.
* **SadTalker** (Apache-2.0): face / lip-sync; used automatically when EchoMimicV2 fails (e.g. out of GPU memory).

1. Open it in Colab: <https://colab.research.google.com/github/kamalnathmurugan14/videoai/blob/main/colab/VideoAI_GPU_worker.ipynb>
2. *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all* (first run downloads ~15 GB, ~20–30 min).
3. Paste the printed **URL** and **TOKEN** into VideoAI → **System check → GPU worker** → *Connect*.
4. Editor → *Look & feel* → *Animation engine* → **HeyGen-like (EchoMimicV2, Colab GPU)** → Render. Keep the Colab tab open.

The notebook embeds `colab/gpu_worker.py` (regenerate with `python colab/build_notebook.py` after editing it). It is a
small FastAPI server behind a free Cloudflare quick tunnel; every request needs the token. Speed on the free T4 is slow
(EchoMimicV2 roughly 10–30 min per 10 s of video, SadTalker ~1 min); free Colab sessions also time out, after which you
run the notebook again and paste the new URL. If the worker is unreachable or fails, that scene falls back to the
built-in engine and the video page says so. EchoMimicV2 drives the hands from its bundled demo pose sequences, so a scene
can be at most as long as the pose sequence (~10+ s); longer scenes use SadTalker.

### Original-photo framing (HeyGen style)

Uploading a photo now defaults to HeyGen's look: your original photo with its own background fills a 9:16 frame and is
animated in place; captions, intro and drawn hands are off (turn them on in *Look & feel*). Switch *Framing* to
*Cut-out on a background* for the scenery look. Landscape frames fit the photo to the height with a blurred fill.

### Realistic lip-sync with SadTalker on this PC (optional)

The built-in engine animates the photo in 2D (fast, any computer). For neural, audio-driven lip-sync, expressions and head
motion, install [SadTalker](https://github.com/OpenTalker/SadTalker) (Apache-2.0):

```powershell
# Windows, from the project folder (needs git, FFmpeg and ~6 GB of disk):
.\setup-sadtalker.ps1          # NVIDIA GPU detected automatically; add -Cpu to force the CPU build
```
The script installs Python 3.10 (SadTalker does not support newer Python), clones SadTalker next to this project into its
own virtualenv with pinned dependencies, downloads the weights, **runs a real self-test**, and writes `SADTALKER_DIR`,
`SADTALKER_PYTHON`, `SADTALKER_ENHANCER` to `backend/.env`. Restart the backend, check **System check → Test SadTalker**,
then pick *Look & feel → Animation engine → Realistic lip-sync (SadTalker)* per video. Each scene's photo is animated by
SadTalker and cut out again per frame, so scenery, captions, intro, music and gestures still apply. If SadTalker fails for
a scene, that scene falls back to the built-in engine and the reason is shown on the video page.

Speed: roughly 1 minute per 10 s of video on a mid-range NVIDIA GPU; on CPU expect 10+ minutes per 10 s.
Quality is a clear step up from the built-in engine, but still below commercial tools such as HeyGen.

**Plugging in another lip-sync model:** avatar animation sits behind `AvatarProvider` (`services/avatar.py`).
The default `amplitude` provider animates the mouth from audio loudness. To use SadTalker / Wav2Lip, install the model
yourself and set:
```
AVATAR_PROVIDER=command
AVATAR_COMMAND=python /opt/sadtalker/inference.py --source_image {image} --driven_audio {audio} --result_video {out}
```
`{image}`, `{audio}` (WAV) and `{out}` (MP4) are substituted; frames are read back and composited. Or subclass
`AvatarProvider` for in-process models.

## Known limitations

- **Animation is a 2D puppet, not a neural model.** The mouth opens with the audio loudness (not phoneme/viseme-accurate),
  and head/body/eye motion is procedural. Hand gestures are *drawn* illustrated hands (skin tone sampled from the photo, sleeve colour from the clothing) that
  point/count/explain per spoken phrase; they are not the person's real hands and not pose-tracked (toggle them off in
  Look & feel). Rendering with gestures takes roughly twice as long. Photo animation locates the face with Google's MediaPipe
  BlazeFace + 478-point Face Mesh models (Apache-2.0 TFLite files in `backend/assets/models`, run locally via
  `ai-edge-litert`), which gives the exact lip seam, mouth corners, eyes and jaw for any pose; tilted photos are rotated
  upright first, tiny faces in wide shots are found by scanning tiles / zooming in. If that runtime is unavailable it
  falls back to OpenCV Haar cascades + proportions (less precise); when no face is found, or the head is strongly
  turned/tilted, the editor warns you. It works best with a clear, front-facing photo. The mouth position comes from the detected face
  box; use the **Fine-tune the mouth** sliders in the Avatar tab to line it up exactly with the lip seam. Background removal uses MediaPipe's person-segmentation model (keeps
  hair, shoulders and clothes, removes busy backgrounds and other people), sharpened with GrabCut; untick
  "Remove photo background" to keep the original. Smiling photos (teeth showing) are first warped to a closed-lip smile
  so the animation opens the mouth naturally along the real lip curve. Plug in SadTalker/Wav2Lip via
  `AVATAR_PROVIDER=command` for true lip-sync.
- **Voices need internet** for edge-tts (best quality, picks the exact voice). If it fails the app falls back to your
  computer's own voices via pyttsx3 (Windows SAPI5 / macOS; picks a male or female voice), then Google TTS (a single
  female voice, so male voices cannot be honoured), then espeak-ng. A warning is shown on the video page and in the voice
  preview whenever a fallback was used. The pyttsx3 path was only exercised on Linux (espeak backend) while developing.
- Caption word timing is estimated from word length across the speech span, not forced-aligned. Non-Latin scripts need a
  suitable system font (Noto fonts are installed in the Docker image); complex-script shaping (Indic, Arabic) is basic.
- Progress is polling-only (no SSE). The in-process worker renders one video at a time.
- Uploaded/rendered files are served from `/storage` by unguessable names without auth (like many CDN-style setups);
  put an authenticated proxy in front if your content is sensitive. No rate limiting on login, no email verification or
  password reset, and guest accounts are never cleaned up.
- Only image media per scene (no video clips). `docker compose` files are provided but were not run in the authoring
  environment (no Docker daemon) — the Postgres path was verified against a local PostgreSQL 16 instead.
