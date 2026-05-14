from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from grade_system.config import Settings, load_settings
from grade_system.persistence.base import Base


@lru_cache(maxsize=None)
def _engine_for_url(database_url: str, echo: bool) -> Engine:
    return create_engine(
        database_url,
        echo=echo,
        future=True,
        pool_pre_ping=True,
    )


def get_engine(settings: Settings | None = None) -> Engine:
    resolved_settings = settings or load_settings()
    return _engine_for_url(resolved_settings.database_url, resolved_settings.sqlalchemy_echo)


def create_session(settings: Settings | None = None) -> Session:
    resolved_settings = settings or load_settings()
    factory = sessionmaker(
        bind=get_engine(resolved_settings),
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    return factory()


def init_database_schema(settings: Settings | None = None) -> None:
    import grade_system.persistence.models  # noqa: F401

    Base.metadata.create_all(bind=get_engine(settings))


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    session = create_session(settings)
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_session() -> Iterator[Session]:
    session = create_session()
    try:
        yield session
    finally:
        session.close()
