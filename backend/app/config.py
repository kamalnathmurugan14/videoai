from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"
    database_url: str = "sqlite:///./storage/videoai.db"
    secret_key: str = "dev-insecure-change-me"
    access_token_minutes: int = 60 * 24 * 7
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    trust_proxy: bool = False  # honour X-Forwarded-For for per-IP limits

    storage_dir: Path = Path("./storage")
    max_upload_mb: int = 8
    max_video_upload_mb: int = 200  # video avatars (10-20 s clips from a phone)

    # limits
    render_daily_limit_user: int = 10
    render_daily_limit_ip: int = 25
    max_scenes: int = 20
    max_scene_chars: int = 1000
    max_total_chars: int = 6000

    # worker / render
    worker_enabled: bool = True
    worker_poll_seconds: float = 1.0
    render_fps: int = 24
    render_scale: float = 1.0  # 1.0 => 1280x720 landscape; lower for faster renders
    ffmpeg_bin: str = "ffmpeg"
    tts_providers: str = "edge,pyttsx3,gtts,espeak"  # tried in order; "synthetic" is a test-only fallback
    tts_timeout_seconds: int = 45
    avatar_provider: str = "amplitude"  # amplitude | command
    avatar_command: str = ""  # e.g. 'python sadtalker/inference.py --image {image} --audio {audio} --out {out}'
    caption_font: str = ""
    # SadTalker (optional realistic lip-sync engine; installed by setup-sadtalker.ps1 into its own Python 3.10 venv)
    sadtalker_dir: str = ""  # folder of the SadTalker checkout (contains inference.py and checkpoints/)
    sadtalker_python: str = ""  # python executable of SadTalker's venv
    sadtalker_enhancer: str = ""  # "gfpgan" = sharper faces (slower); empty = off
    sadtalker_size: int = 256  # 256 or 512 (512 needs the 512 checkpoint and more GPU memory)
    sadtalker_timeout_seconds: int = 3600

    # optional LLM for text -> scenes (heuristic summariser is used when unset)
    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-5-5"

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def tts_list(self) -> list[str]:
        return [p.strip().lower() for p in self.tts_providers.split(",") if p.strip()]


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if s.environment == "production" and s.secret_key == "dev-insecure-change-me":
        raise RuntimeError("SECRET_KEY must be set in production")
    return s


ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
APP_VERSION = "1.8.0"  # shown in the UI footer and /api/system so you can confirm which build is running
