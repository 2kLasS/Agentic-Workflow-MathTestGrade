from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from grade_system.persistence.base import Base, TimestampMixin


class GradingTaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class GradingTask(TimestampMixin, Base):
    __tablename__ = "grading_tasks"
    __table_args__ = (
        Index("uk_grading_tasks_task_id", "task_id", unique=True),
        Index("idx_grading_tasks_user_created_at", "user_id", "created_at"),
        Index("idx_grading_tasks_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(36), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16),
        default=GradingTaskStatus.PENDING.value,
        nullable=False,
    )
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    student_answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    question_excerpt: Mapped[str] = mapped_column(String(255), nullable=False)
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    attribution_summary_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    error_message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    workflow_version: Mapped[str] = mapped_column(String(32), nullable=False)
    llm_call_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    reasoning_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    request_snapshot_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    result_snapshot_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="grading_tasks")
