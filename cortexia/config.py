"""Application configuration via environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration loaded from environment / .env file."""

    # Application
    app_name: str = "cortexia"
    app_env: Literal["development", "staging", "production"] = "production"
    log_level: str = "INFO"
    debug: bool = False
    secret_key: str = ""  # must be set via SECRET_KEY env var
    api_key: str = ""  # must be set via API_KEY env var

    # Database
    database_url: str = (
        "postgresql+asyncpg://cortexia:password@localhost:5432/cortexia"
    )
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            # Handle JSON arrays
            if v.startswith("["):
                import json

                return json.loads(v)
            # Handle comma-separated strings
            return [origin.strip() for origin in v.split(",")]
        return v

    # ML Models
    model_backend: str = "retinaface"
    model_cache_dir: Path = Path("/app/models")
    embedding_dim: int = 512
    detection_threshold: float = 0.5

    # Trust Pipeline
    trust_pipeline_enabled: bool = True
    antispoof_enabled: bool = True
    antispoof_threshold: float = 0.7
    recognition_threshold: float = 0.45
    unknown_threshold: float = 0.35

    # Trust score weights (must sum to 1.0)
    trust_weight_detection: float = 0.20
    trust_weight_liveness: float = 0.40
    trust_weight_recognition: float = 0.40

    # Face Attributes
    attributes_enabled: bool = True
    age_estimation: bool = True
    gender_estimation: bool = True
    emotion_estimation: bool = True

    # Streaming
    max_faces_per_frame: int = 20
    stream_fps: int = 15
    tracker_max_age: int = 30
    recognition_interval_frames: int = 5

    # Rate Limiting
    rate_limit_per_minute: int = 60

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
