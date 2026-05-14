from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from grade_system.api.routers import auth, grading_tasks, health
from grade_system.config import load_settings
from grade_system.persistence import create_session, init_database_schema
from grade_system.persistence.repositories.grading_task_repository import (
    GradingTaskRepository,
)


def create_app() -> FastAPI:
    settings = load_settings()

    app = FastAPI(
        title="GradeSystem API",
        version="0.1.0",
    )

    if settings.backend_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.backend_cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.on_event("startup")
    def startup_event() -> None:
        if settings.auto_create_tables:
            init_database_schema(settings)

        try:
            session = create_session(settings)
            try:
                repository = GradingTaskRepository(session)
                repaired_count = repository.mark_all_running_as_failed(
                    "Task interrupted before completion."
                )
                if repaired_count:
                    session.commit()
                else:
                    session.rollback()
            finally:
                session.close()
        except Exception:
            # Do not block service startup on best-effort recovery.
            pass

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(grading_tasks.router)
    return app


app = create_app()
