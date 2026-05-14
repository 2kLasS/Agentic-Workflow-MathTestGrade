from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import text

from grade_system.persistence import create_session

router = APIRouter(tags=["health"])


@router.get("/api/v1/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/api/v1/ready")
def readiness_check() -> dict[str, str]:
    try:
        session = create_session()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"数据库配置或连接初始化失败: {exc}",
        ) from exc

    try:
        session.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"数据库未就绪: {exc}",
        ) from exc
    finally:
        session.close()
