from pydantic import BaseModel, Field, HttpUrl


class VoicePreview(BaseModel):
    voice: str = Field(pattern=r"^[A-Za-z0-9-]{2,64}$")
    text: str = Field(default="Hello! This is how I will sound in your video.", min_length=1, max_length=200)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class ImportText(BaseModel):
    text: str = Field(min_length=20, max_length=20000)
    max_scenes: int = Field(default=6, ge=1, le=12)


class ImportUrl(BaseModel):
    url: HttpUrl
    max_scenes: int = Field(default=6, ge=1, le=12)
