from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from grade_system.persistence.models import GradingTask, GradingTaskStatus


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GradingTaskRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        *,
        user_id: int,
        question_text: str,
        student_answer_text: str,
        question_excerpt: str,
        request_snapshot_json: dict[str, Any],
        workflow_version: str,
    ) -> GradingTask:
        task = GradingTask(
            task_id=str(uuid4()),
            user_id=user_id,
            status=GradingTaskStatus.PENDING.value,
            question_text=question_text,
            student_answer_text=student_answer_text,
            question_excerpt=question_excerpt,
            request_snapshot_json=request_snapshot_json,
            workflow_version=workflow_version,
        )
        self.session.add(task)
        return task

    def get_by_task_id(self, task_id: str) -> GradingTask | None:
        return self.session.scalar(select(GradingTask).where(GradingTask.task_id == task_id))

    def get_for_user(self, *, task_id: str, user_id: int) -> GradingTask | None:
        return self.session.scalar(
            select(GradingTask).where(
                GradingTask.task_id == task_id,
                GradingTask.user_id == user_id,
            )
        )

    def list_for_user(
        self,
        *,
        user_id: int,
        page: int,
        page_size: int,
    ) -> tuple[list[GradingTask], int]:
        offset = max(page - 1, 0) * page_size
        query = (
            select(GradingTask)
            .where(GradingTask.user_id == user_id)
            .order_by(GradingTask.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        total = self.session.scalar(
            select(func.count()).select_from(GradingTask).where(GradingTask.user_id == user_id)
        )
        items = list(self.session.scalars(query).all())
        return items, int(total or 0)

    def mark_running(self, task: GradingTask) -> None:
        task.status = GradingTaskStatus.RUNNING.value
        task.started_at = utcnow()
        task.error_message = ""

    def mark_succeeded(
        self,
        task: GradingTask,
        *,
        final_output: dict[str, Any],
        is_correct: bool,
        attribution_summary_text: str,
        usage: dict[str, int],
    ) -> None:
        task.status = GradingTaskStatus.SUCCEEDED.value
        task.is_correct = is_correct
        task.attribution_summary_text = attribution_summary_text
        task.result_snapshot_json = final_output
        task.error_message = ""
        task.finished_at = utcnow()
        task.llm_call_count = int(usage.get("llm_call_count", 0))
        task.input_tokens = int(usage.get("input_tokens", 0))
        task.output_tokens = int(usage.get("output_tokens", 0))
        task.reasoning_tokens = int(usage.get("reasoning_tokens", 0))
        task.total_tokens = int(usage.get("total_tokens", 0))

    def mark_failed(self, task: GradingTask, error_message: str) -> None:
        task.status = GradingTaskStatus.FAILED.value
        task.error_message = error_message
        task.finished_at = utcnow()

    def mark_all_running_as_failed(self, error_message: str) -> int:
        statement = (
            update(GradingTask)
            .where(GradingTask.status == GradingTaskStatus.RUNNING.value)
            .values(
                status=GradingTaskStatus.FAILED.value,
                error_message=error_message,
                finished_at=utcnow(),
            )
        )
        result = self.session.execute(statement)
        return int(result.rowcount or 0)
