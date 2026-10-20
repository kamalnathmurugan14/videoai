# VideoAI — free AI avatar video creator

A free, self-hostable alternative to HeyGen / Vidnoz. Write a multi-scene script, pick an animated avatar and a voice,
and render a captioned MP4 — no paid APIs required.

**Stack:** React 19 + Vite + React Router + Tailwind · FastAPI (Python 3.11+) · SQLAlchemy 2 + Alembic ·
SQLite (default) or PostgreSQL · FFmpeg · edge-tts / gTTS / espeak-ng · Pillow + NumPy.

## Features

- **Auth:** email/password (bcrypt + JWT) and one-click **guest mode**.
- **Dashboard:** list, create (blank or from template), rename, duplicate, delete, status
  (`draft → queued → rendering → done / failed`), preview player, MP4 download.
- **Editor:** 6 bundled avatars + upload your own image · multi-scene script (per-scene text, background, image) ·
  voice & language picker (24 voices), speed, voice preview · colour / gradient / image backgrounds ·
  auto-captions (3 styles, word highlight) · 16:9 / 9:16 / 1:1 · templates (explainer, social promo, product intro) ·
  background music (3 procedural royalty-free loops) · text / blog-URL → scenes.
- **Render pipeline:** DB-backed job queue + background worker → per-scene TTS → amplitude-driven talking avatar →
  captions → FFmpeg encode → concat (+ music) → MP4 + thumbnail. Progress via polling.
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

**Plugging in a real lip-sync model:** avatar animation sits behind `AvatarProvider` (`services/avatar.py`).
The default `amplitude` provider animates the mouth from audio loudness. To use SadTalker / Wav2Lip, install the model
yourself and set:
```
AVATAR_PROVIDER=command
AVATAR_COMMAND=python /opt/sadtalker/inference.py --source_image {image} --driven_audio {audio} --result_video {out}
```
`{image}`, `{audio}` (WAV) and `{out}` (MP4) are substituted; frames are read back and composited. Or subclass
`AvatarProvider` for in-process models.

## Known limitations

- **Lip-sync is amplitude-based**, not phoneme/viseme-accurate, unless you plug in a model provider. Bundled avatars are
  flat-style illustrations. For uploaded photos the jaw is stretched and a dark mouth opening drawn at an assumed mouth
  position (≈2/3 down the image) — results vary by photo.
- **Voices need internet** for edge-tts (best quality) and gTTS; without it the offline `espeak-ng` voice (robotic) is used.
  The job's detail page notes which engine was used. Non-English voices on the espeak fallback depend on installed espeak data.
- Caption word timing is estimated from word length across the speech span, not forced-aligned. Non-Latin scripts need a
  suitable system font (Noto fonts are installed in the Docker image); complex-script shaping (Indic, Arabic) is basic.
- Progress is polling-only (no SSE). The in-process worker renders one video at a time.
- Uploaded/rendered files are served from `/storage` by unguessable names without auth (like many CDN-style setups);
  put an authenticated proxy in front if your content is sensitive. No rate limiting on login, no email verification or
  password reset, and guest accounts are never cleaned up.
- Only image media per scene (no video clips). `docker compose` files are provided but were not run in the authoring
  environment (no Docker daemon) — the Postgres path was verified against a local PostgreSQL 16 instead.
