from __future__ import annotations

from typing import Any, TypedDict


class WorkflowState(TypedDict, total=False):
    question_text: str
    student_answer_text: str
    is_multi_part: bool
    shared_stem_text: str
    subproblems: list[str]
    student_subanswers: list[str]
    shared_problem_context: list[str]

    current_student_state: list[str]
    current_correct_state: list[str]
    history_has_error: bool

    current_subproblem_index: int
    current_subproblem_text: str
    current_student_subanswer: str
    current_student_steps: list[str]
    current_step_index: int
    current_step_text: str
    step_intent: str
    current_step_status: str
    current_step_judge_reason: str
    step_records: list[dict[str, Any]]
    completion_is_complete: bool
    current_is_correct: bool
    math_tool_request: dict[str, Any] | None
    math_tool_result: dict[str, Any] | None
    correct_solution_started: bool
    correct_solution_steps: list[str]
    correct_solution_cursor: int

    subproblem_results: list[dict[str, Any]]
    is_correct: bool
