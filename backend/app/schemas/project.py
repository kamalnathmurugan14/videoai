from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Aspect = Literal["16:9", "9:16", "1:1"]
CaptionStyle = Literal["classic", "bold", "minimal"]
HexColor = str


class Background(BaseModel):
    type: Literal["color", "gradient", "image", "scenery"] = "gradient"
    color: str = Field(default="#4f46e5", pattern=r"^#[0-9a-fA-F]{6}$")
    color2: str = Field(default="#9333ea", pattern=r"^#[0-9a-fA-F]{6}$")
    asset_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    scenery_id: str | None = Field(default=None, pattern=r"^[a-z0-9_-]{1,32}$")


class SceneIn(BaseModel):
    text: str = Field(default="", max_length=2000)
    background: Background | None = None  # None => inherit project background
    media_asset_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


class SceneOut(SceneIn):
    id: int
    position: int

    model_config = {"from_attributes": True}


class ProjectCreate(BaseModel):
    title: str = Field(default="Untitled video", min_length=1, max_length=120)
    template_id: str | None = None


class ProjectUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    aspect_ratio: Aspect | None = None
    avatar_id: str | None = Field(default=None, pattern=r"^([a-z0-9_-]{1,32}|asset:[0-9a-f]{32})$")
    voice: str | None = Field(default=None, pattern=r"^[A-Za-z0-9-]{2,64}$")
    speed: float | None = Field(default=None, ge=0.5, le=2.0)
    captions_enabled: bool | None = None
    caption_style: CaptionStyle | None = None
    music_id: str | None = Field(default=None, pattern=r"^[a-z0-9_-]{1,32}$")
    clear_music: bool = False
    music_volume: float | None = Field(default=None, ge=0.0, le=1.0)
    background: Background | None = None
    intro_enabled: bool | None = None
    intro_title: str | None = Field(default=None, max_length=120)
    gestures_enabled: bool | None = None
    animation_engine: Literal["builtin", "sadtalker", "gpu_echomimic", "gpu_sadtalker", "gpu_lipsync"] | None = None
    framing: Literal["scene", "photo"] | None = None
    scenes: list[SceneIn] | None = None

    @field_validator("title")
    @classmethod
    def strip_title(cls, v):
        v = v.strip() if v else v
        if not v:
            raise ValueError("title cannot be blank")
        return v


class JobOut(BaseModel):
    id: int
    project_id: int
    status: str
    progress: int
    stage: str
    info: str
    error: str | None
    created_at: datetime
    finished_at: datetime | None

    model_config = {"from_attributes": True}


class ProjectSummary(BaseModel):
    id: int
    title: str
    status: str
    aspect_ratio: str
    duration: float | None
    video_url: str | None
    thumbnail_url: str | None
    scene_count: int
    created_at: datetime
    updated_at: datetime


class ProjectOut(ProjectSummary):
    avatar_id: str
    voice: str
    speed: float
    captions_enabled: bool
    caption_style: str
    music_id: str | None
    music_volume: float
    background: Background
    intro_enabled: bool
    intro_title: str | None
    gestures_enabled: bool
    animation_engine: str
    framing: str
    error: str | None
    scenes: list[SceneOut]
    latest_job: JobOut | None
