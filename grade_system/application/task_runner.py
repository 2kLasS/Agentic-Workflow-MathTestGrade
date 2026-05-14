from __future__ import annotations

from grade_system.application.grading_service import GradingTaskService
from grade_system.persistence.repositories.grading_task_repository import (
    GradingTaskRepository,
)
from grade_system.persistence.session import session_scope


def run_grading_task(task_id: str) -> None:
    with session_scope() as session:
        repository = GradingTaskRepository(session)
        task = repository.get_by_task_id(task_id)
        if task is None:
            return
        GradingTaskService().run_task(session, task)
