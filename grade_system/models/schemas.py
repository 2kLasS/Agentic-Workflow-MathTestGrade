from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GradeWorkflowInput(BaseModel):
    question_text: str = Field(description="Full problem text")
    student_answer_text: str = Field(description="Full student answer text")


class ProblemParseResult(BaseModel):
    is_multi_part: bool
    shared_stem_text: str = Field(default="")
    subproblems: list[str] = Field(default_factory=list)
    student_subanswers: list[str] = Field(default_factory=list)


class ProblemReviewResult(BaseModel):
    shared_problem_context: list[str] = Field(default_factory=list)


class StepSplitResult(BaseModel):
    student_steps: list[str] = Field(default_factory=list)


class MathToolRequest(BaseModel):
    mode: str
    payload: dict[str, Any] = Field(default_factory=dict)


class StepAnalysisResult(BaseModel):
    step_intent: str = Field(default="")
    tool_needed: bool
    math_tool_request: MathToolRequest | None = Field(default=None)


class StepJudgeResult(BaseModel):
    step_status: str = Field(default="correct")
    judge_reason: str = Field(default="")
    student_facts_update: list[str] = Field(default_factory=list)


class CorrectAlignmentResult(BaseModel):
    aligned_correct_facts: list[str] = Field(default_factory=list)


class CompletionCheckResult(BaseModel):
    is_complete: bool
    issue_reason: str = Field(default="")


class CorrectSolutionResult(BaseModel):
    correct_solution_process: list[str] = Field(default_factory=list)
    correct_solution_steps: list[str] = Field(default_factory=list)


class AttributionResult(BaseModel):
    primary_reason_type: str = Field(default="")
    reason_type: list[str] = Field(default_factory=list)
    attribution_reason: str = Field(default="")
    feedback_text: str = Field(default="")


class StepRecord(BaseModel):
    step_index: int
    step_text: str = Field(default="")
    step_intent: str = Field(default="")
    step_status: str = Field(default="correct")
    judge_reason: str = Field(default="")
    student_state_before: list[str] = Field(default_factory=list)
    correct_state_before: list[str] = Field(default_factory=list)
    student_state_after: list[str] = Field(default_factory=list)
    correct_state_after: list[str] = Field(default_factory=list)


class ErrorReport(BaseModel):
    step_index: int
    step_text: str = Field(default="")
    judge_reason: str = Field(default="")
    primary_reason_type: str = Field(default="")
    reason_type: list[str] = Field(default_factory=list)
    attribution_reason: str = Field(default="")
    feedback_text: str = Field(default="")


class SubproblemResult(BaseModel):
    subproblem_index: int
    is_correct: bool
    student_steps: list[str] = Field(default_factory=list)
    step_records: list[StepRecord] = Field(default_factory=list)
    correct_solution_steps: list[str] = Field(default_factory=list)
    error_reports: list[ErrorReport] = Field(default_factory=list)


class GradeWorkflowOutput(BaseModel):
    is_correct: bool
    subproblem_results: list[SubproblemResult] = Field(default_factory=list)
