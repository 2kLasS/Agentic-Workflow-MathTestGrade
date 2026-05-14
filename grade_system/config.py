from __future__ import annotations

import os
from functools import lru_cache
from urllib.parse import quote_plus

from pydantic import BaseModel, Field


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> list[str]:
    raw_value = os.getenv(name, "")
    return [item.strip() for item in raw_value.split(",") if item.strip()]


class Settings(BaseModel):
    qwen_api_key: str = Field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", "")
    )
    qwen_base_url: str = Field(
        default_factory=lambda: os.getenv("QWEN_BASE_URL", "")
    )
    qwen_model: str = Field(
        default_factory=lambda: os.getenv("QWEN_MODEL", "qwen3.5-plus")
    )
    qwen_temperature: float = Field(
        default_factory=lambda: float(os.getenv("QWEN_TEMPERATURE", "0.5"))
    )
    qwen_request_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("QWEN_TIMEOUT_SECONDS", "300"))
    )
    qwen_max_retries: int = Field(
        default_factory=lambda: int(os.getenv("QWEN_MAX_RETRIES", "0"))
    )

    mysql_url: str = Field(
        default_factory=lambda: os.getenv("MYSQL_URL")
        or os.getenv("DATABASE_URL", "")
    )
    mysql_host: str = Field(default_factory=lambda: os.getenv("MYSQL_HOST", "localhost"))
    mysql_port: int = Field(default_factory=lambda: int(os.getenv("MYSQL_PORT", "3306")))
    mysql_database: str = Field(
        default_factory=lambda: os.getenv("MYSQL_DATABASE", "grade_system")
    )
    mysql_user: str = Field(default_factory=lambda: os.getenv("MYSQL_USER", ""))
    mysql_password: str = Field(
        default_factory=lambda: os.getenv("MYSQL_PASSWORD", "")
    )
    sqlalchemy_echo: bool = Field(
        default_factory=lambda: _env_bool("SQLALCHEMY_ECHO", False)
    )
    auto_create_tables: bool = Field(
        default_factory=lambda: _env_bool("AUTO_CREATE_TABLES", False)
    )

    jwt_secret_key: str = Field(
        default_factory=lambda: os.getenv("JWT_SECRET_KEY", "")
    )
    jwt_algorithm: str = Field(
        default_factory=lambda: os.getenv("JWT_ALGORITHM", "HS256")
    )
    jwt_access_token_expire_minutes: int = Field(
        default_factory=lambda: int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    )
    jwt_refresh_token_expire_days: int = Field(
        default_factory=lambda: int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    )

    backend_cors_origins: list[str] = Field(
        default_factory=lambda: _env_list("BACKEND_CORS_ORIGINS")
    )
    grading_workflow_version: str = Field(
        default_factory=lambda: os.getenv("GRADING_WORKFLOW_VERSION", "workflow-v1")
    )

    @property
    def database_url(self) -> str:
        if self.mysql_url:
            return self.mysql_url

        auth_part = ""
        if self.mysql_user and self.mysql_password:
            auth_part = f"{self.mysql_user}:{quote_plus(self.mysql_password)}"
        elif self.mysql_user:
            auth_part = self.mysql_user
        if auth_part:
            auth_part = f"{auth_part}@"

        return (
            f"mysql+pymysql://{auth_part}{self.mysql_host}:{self.mysql_port}/"
            f"{self.mysql_database}?charset=utf8mb4"
        )


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    return Settings()
