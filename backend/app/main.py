import logging
from contextlib import asynccontextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import auth, catalog as catalog_api, media, projects
from .config import ASSETS_DIR, get_settings
from .services import music, storage
from .workers.worker import WorkerThread

logging.basicConfig(level=logging.INFO)


def run_migrations() -> None:
    cfg = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
    cfg.set_main_option("script_location", str(Path(__file__).resolve().parent.parent / "alembic"))
    command.upgrade(cfg, "head")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    for d in ("uploads", "renders", "thumbs", "tmp", "previews", "music"):
        storage.subdir(d)
    run_migrations()
    music.ensure_library()
    worker = None
    if s.worker_enabled:
        worker = WorkerThread()
        worker.start()
    yield
    if worker:
        worker.stop_event.set()


app = FastAPI(title="VideoAI", version="1.0.0", description="Free AI avatar video creator", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_list, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

for r in (auth.router, projects.router, media.router, catalog_api.router):
    app.include_router(r, prefix="/api")


@app.get("/api/health", tags=["meta"])
def health():
    return {"status": "ok"}


get_settings().storage_dir.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(get_settings().storage_dir)), name="storage")
app.mount("/avatar-art", StaticFiles(directory=str(ASSETS_DIR / "avatars")), name="avatar-art")
app.mount("/scenery-art", StaticFiles(directory=str(ASSETS_DIR / "sceneries")), name="scenery-art")
