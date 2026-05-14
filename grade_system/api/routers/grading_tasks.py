from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from grade_system.api import deps
from grade_system.api.schemas.grading import (
    CreateGradingTaskRequest,
    CreateGradingTaskResponse,
    GradingTaskDetailResponse,
    GradingTaskListItemResponse,
    GradingTaskListResponse,
    TokenUsageResponse,
)
from grade_system.application.grading_service import GradingTaskService
from grade_system.application.task_runner import run_grading_task
from grade_system.persistence import get_db_session
from grade_system.persistence.models import User
from grade_system.persistence.repositories.grading_task_repository import (
    GradingTaskRepository,
)

router = APIRouter(prefix="/api/v1/grading-tasks", tags=["grading-tasks"])


@router.post("", response_model=CreateGradingTaskResponse, status_code=status.HTTP_202_ACCEPTED)
def create_grading_task(
    payload: CreateGradingTaskRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_db_session),
    current_user: User = Depends(deps.get_current_user),
) -> CreateGradingTaskResponse:
    service = GradingTaskService()
    task = service.create_task(
        session,
        user_id=current_user.id,
        question_text=payload.question_text,
        student_answer_text=payload.student_answer_text,
    )
    session.commit()
    session.refresh(task)
    background_tasks.add_task(run_grading_task, task.task_id)
    return CreateGradingTaskResponse(
        task_id=task.task_id,
        status=task.status,
        created_at=task.created_at,
    )


@router.get("", response_model=GradingTaskListResponse)
def list_grading_tasks(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_db_session),
    current_user: User = Depends(deps.get_current_user),
) -> GradingTaskListResponse:
    repository = GradingTaskRepository(session)
    items, total = repository.list_for_user(
        user_id=current_user.id,
        page=page,
        page_size=page_size,
    )
    return GradingTaskListResponse(
        items=[
            GradingTaskListItemResponse(
                task_id=item.task_id,
                question_excerpt=item.question_excerpt,
                status=item.status,
                is_correct=item.is_correct,
                created_at=item.created_at,
            )
            for item in items
        ],
        page=page,
        page_size=page_size,
        total=total,
    )


@router.get("/{task_id}", response_model=GradingTaskDetailResponse)
def get_grading_task_detail(
    task_id: str,
    session: Session = Depends(get_db_session),
    current_user: User = Depends(deps.get_current_user),
) -> GradingTaskDetailResponse:
    repository = GradingTaskRepository(session)
    task = repository.get_for_user(task_id=task_id, user_id=current_user.id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在。")

    return GradingTaskDetailResponse(
        task_id=task.task_id,
        status=task.status,
        question_text=task.question_text,
        student_answer_text=task.student_answer_text,
        is_correct=task.is_correct,
        attribution_summary_text=task.attribution_summary_text,
        created_at=task.created_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
        usage=TokenUsageResponse(
            llm_call_count=task.llm_call_count,
            input_tokens=task.input_tokens,
            output_tokens=task.output_tokens,
            reasoning_tokens=task.reasoning_tokens,
            total_tokens=task.total_tokens,
        ),
        result=task.result_snapshot_json,
        error_message=task.error_message,
    )
