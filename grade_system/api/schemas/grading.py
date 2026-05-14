from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CreateGradingTaskRequest(BaseModel):
    question_text: str = Field(min_length=1)
    student_answer_text: str = Field(min_length=1)


class CreateGradingTaskResponse(BaseModel):
    task_id: str
    status: str
    created_at: datetime


class GradingTaskListItemResponse(BaseModel):
    task_id: str
    question_excerpt: str
    status: str
    is_correct: bool | None = None
    created_at: datetime


class GradingTaskListResponse(BaseModel):
    items: list[GradingTaskListItemResponse]
    page: int
    page_size: int
    total: int


class TokenUsageResponse(BaseModel):
    llm_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


class GradingTaskDetailResponse(BaseModel):
    task_id: str
    status: str
    question_text: str
    student_answer_text: str
    is_correct: bool | None = None
    attribution_summary_text: str = ""
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    usage: TokenUsageResponse
    result: dict[str, Any] | None = None
    error_message: str = ""
