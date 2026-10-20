from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..db import Base
from .base import utcnow


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(20), default="draft")  # draft|queued|rendering|done|failed
    aspect_ratio: Mapped[str] = mapped_column(String(8), default="16:9")
    avatar_id: Mapped[str] = mapped_column(String(64), default="aria")  # builtin id or "asset:<id>"
    voice: Mapped[str] = mapped_column(String(64), default="en-US-AriaNeural")
    speed: Mapped[float] = mapped_column(Float, default=1.0)
    captions_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    caption_style: Mapped[str] = mapped_column(String(20), default="bold")
    music_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    music_volume: Mapped[float] = mapped_column(Float, default=0.15)
    background: Mapped[dict] = mapped_column(JSON, default=dict)
    intro_enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default=sa.true())
    intro_title: Mapped[str | None] = mapped_column(String(120), nullable=True)
    gestures_enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default=sa.true())
    animation_engine: Mapped[str] = mapped_column(String(16), default="builtin", server_default="builtin")
    framing: Mapped[str] = mapped_column(String(10), default="scene", server_default="scene")
    video_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    thumbnail_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    scenes: Mapped[list["Scene"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="Scene.position"
    )
    jobs: Mapped[list["RenderJob"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="RenderJob.id"
    )


class Scene(Base):
    __tablename__ = "scenes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text, default="")
    background: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # None => inherit project background
    media_asset_id: Mapped[str | None] = mapped_column(String(32), nullable=True)

    project: Mapped[Project] = relationship(back_populates="scenes")


class RenderJob(Base):
    __tablename__ = "render_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    ip: Mapped[str] = mapped_column(String(64), default="", index=True)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)  # queued|running|done|failed
    progress: Mapped[int] = mapped_column(Integer, default=0)
    stage: Mapped[str] = mapped_column(String(120), default="Queued")
    info: Mapped[str] = mapped_column(Text, default="")  # notes e.g. TTS fallbacks
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    project: Mapped[Project] = relationship(back_populates="jobs")
