from __future__ import annotations

from copy import deepcopy
from typing import Any

from grade_system.models.schemas import (
    AttributionResult,
    CompletionCheckResult,
    CorrectAlignmentResult,
    CorrectSolutionResult,
    ErrorReport,
    ProblemParseResult,
    ProblemReviewResult,
    StepAnalysisResult,
    StepJudgeResult,
    StepRecord,
    StepSplitResult,
    SubproblemResult,
)
from grade_system.models.state import WorkflowState
from grade_system.prompts import (
    build_attribution_prompt,
    build_completion_prompt,
    build_correct_alignment_prompt,
    build_correct_solution_prompt,
    build_parse_problem_prompt,
    build_problem_review_prompt,
    build_step_analysis_prompt,
    build_step_judge_prompt,
    build_step_split_prompt,
)
from grade_system.services.llm_service import QwenWorkflowLLM
from grade_system.tools.math_tools import MathToolEngine


class WorkflowNodes:
    def __init__(self, llm: QwenWorkflowLLM, math_tool: MathToolEngine) -> None:
        self.llm = llm
        self.math_tool = math_tool

    def parse_problem(self, state: WorkflowState) -> dict[str, Any]:
        system_prompt, user_prompt = build_parse_problem_prompt(
            state["question_text"],
            state["student_answer_text"],
        )
        result = self.llm.invoke_structured(ProblemParseResult, system_prompt, user_prompt)

        is_multi_part = result.is_multi_part
        shared_stem_text = (result.shared_stem_text or "").strip()
        subproblems = [item.strip() for item in result.subproblems if item and item.strip()]
        student_subanswers = [item.strip() for item in result.student_subanswers]

        if not is_multi_part:
            return {
                "is_multi_part": False,
                "shared_stem_text": "",
                "subproblems": [state["question_text"]],
                "student_subanswers": [state["student_answer_text"]],
                "history_has_error": False,
                "current_subproblem_index": 0,
                "subproblem_results": [],
            }

        if not subproblems:
            subproblems = [state["question_text"]]
            is_multi_part = False
            shared_stem_text = ""

        if len(student_subanswers) < len(subproblems):
            student_subanswers.extend([""] * (len(subproblems) - len(student_subanswers)))

        if len(student_subanswers) > len(subproblems):
            student_subanswers = student_subanswers[: len(subproblems)]

        return {
            "is_multi_part": is_multi_part if len(subproblems) > 1 else False,
            "shared_stem_text": shared_stem_text if len(subproblems) > 1 else "",
            "subproblems": subproblems,
            "student_subanswers": student_subanswers,
            "history_has_error": False,
            "current_subproblem_index": 0,
            "subproblem_results": [],
        }

    def review_problem(self, state: WorkflowState) -> dict[str, Any]:
        review_source_text = self._build_review_source_text(state)
        system_prompt, user_prompt = build_problem_review_prompt(
            review_source_text,
            state["subproblems"],
            state.get("is_multi_part", False),
        )
        result = self.llm.invoke_structured(ProblemReviewResult, system_prompt, user_prompt)
        shared_problem_context = self._normalize_facts(result.shared_problem_context)
        return {
            "shared_problem_context": shared_problem_context,
            "current_student_state": list(shared_problem_context),
            "current_correct_state": list(shared_problem_context),
        }

    def prepare_subproblem(self, state: WorkflowState) -> dict[str, Any]:
        current_index = state["current_subproblem_index"]
        return {
            "current_subproblem_text": state["subproblems"][current_index],
            "current_student_subanswer": state["student_subanswers"][current_index],
            "current_student_steps": [],
            "current_step_index": 0,
            "current_step_text": "",
            "step_intent": "",
            "current_step_status": "correct",
            "current_step_judge_reason": "",
            "step_records": [],
            "completion_is_complete": True,
            "current_is_correct": True,
            "math_tool_request": None,
            "math_tool_result": None,
            "correct_solution_started": False,
            "correct_solution_steps": [],
            "correct_solution_cursor": 0,
        }

    def split_steps(self, state: WorkflowState) -> dict[str, Any]:
        system_prompt, user_prompt = build_step_split_prompt(
            state.get("shared_stem_text", ""),
            state["current_subproblem_text"],
            state["current_student_subanswer"],
        )
        result = self.llm.invoke_structured(StepSplitResult, system_prompt, user_prompt)
        return {"current_student_steps": result.student_steps}

    def analyze_step(self, state: WorkflowState) -> dict[str, Any]:
        step_index = state["current_step_index"]
        current_step_text = state["current_student_steps"][step_index]
        system_prompt, user_prompt = build_step_analysis_prompt(
            state.get("shared_stem_text", ""),
            state.get("shared_problem_context", []),
            state["current_subproblem_text"],
            current_step_text,
        )
        result = self.llm.invoke_structured(StepAnalysisResult, system_prompt, user_prompt)

        return {
            "current_step_text": current_step_text,
            "step_intent": result.step_intent,
            "math_tool_request": (
                result.math_tool_request.model_dump()
                if result.tool_needed and result.math_tool_request
                else None
            ),
            "math_tool_result": None,
        }

    def run_math_tool(self, state: WorkflowState) -> dict[str, Any]:
        request = state.get("math_tool_request")
        if not request:
            return {"math_tool_result": None}

        return {
            "math_tool_result": self.math_tool.run(
                mode=request["mode"],
                payload=request.get("payload", {}),
            )
        }

    def judge_step(self, state: WorkflowState) -> dict[str, Any]:
        student_state_before = list(state.get("current_student_state", []))
        correct_state_before = list(state.get("current_correct_state", []))

        system_prompt, user_prompt = build_step_judge_prompt(
            state.get("shared_stem_text", ""),
            state.get("shared_problem_context", []),
            state["current_subproblem_text"],
            state["current_step_text"],
            state.get("step_intent", ""),
            state.get("history_has_error", False),
            student_state_before,
            correct_state_before,
            state.get("math_tool_result"),
        )
        result = self.llm.invoke_structured(StepJudgeResult, system_prompt, user_prompt)

        student_facts_update = self._normalize_facts(result.student_facts_update)
        student_state_after = self._merge_facts(student_state_before, student_facts_update)

        correct_state_after = list(correct_state_before)
        if not state.get("correct_solution_started", False) and result.step_status == "correct":
            correct_state_after = self._merge_facts(correct_state_after, student_facts_update)

        step_record = StepRecord(
            step_index=state["current_step_index"] + 1,
            step_text=state["current_step_text"],
            step_intent=state.get("step_intent", ""),
            step_status=result.step_status,
            judge_reason=result.judge_reason,
            student_state_before=student_state_before,
            correct_state_before=correct_state_before,
            student_state_after=student_state_after,
            correct_state_after=correct_state_after,
        )

        updated_step_records = [*state.get("step_records", []), step_record.model_dump()]
        return {
            "current_step_status": result.step_status,
            "current_step_judge_reason": result.judge_reason,
            "current_student_state": student_state_after,
            "current_correct_state": correct_state_after,
            "step_records": updated_step_records,
            "history_has_error": state.get("history_has_error", False)
            or result.step_status != "correct",
            "current_is_correct": state.get("current_is_correct", True)
            and result.step_status == "correct",
        }

    def generate_correct_solution(self, state: WorkflowState) -> dict[str, Any]:
        system_prompt, user_prompt = build_correct_solution_prompt(
            state.get("shared_stem_text", ""),
            state.get("shared_problem_context", []),
            state["current_subproblem_text"],
            state.get("current_correct_state", []),
        )
        result = self.llm.invoke_structured(CorrectSolutionResult, system_prompt, user_prompt)
        correct_solution_steps = self._normalize_facts(result.correct_solution_steps)

        return {
            "correct_solution_started": True,
            "correct_solution_steps": correct_solution_steps,
            "correct_solution_cursor": 0,
        }

    def align_correct_frontier(self, state: WorkflowState) -> dict[str, Any]:
        remaining_correct_solution_steps = state.get("correct_solution_steps", [])[
            state.get("correct_solution_cursor", 0) :
        ]
        if not remaining_correct_solution_steps:
            return self._update_last_step_record(
                state=state,
                current_correct_state=list(state.get("current_correct_state", [])),
            )

        current_step_index = state.get("current_step_index", 0)
        current_student_steps = state.get("current_student_steps", [])
        next_step_text = None
        if current_step_index + 1 < len(current_student_steps):
            next_step_text = current_student_steps[current_step_index + 1]

        system_prompt, user_prompt = build_correct_alignment_prompt(
            state.get("shared_stem_text", ""),
            state["current_subproblem_text"],
            state.get("current_step_text", ""),
            state.get("step_intent", ""),
            next_step_text,
            state.get("current_correct_state", []),
            remaining_correct_solution_steps,
        )
        result = self.llm.invoke_structured(CorrectAlignmentResult, system_prompt, user_prompt)

        aligned_facts = self._normalize_facts(result.aligned_correct_facts)
        accepted_facts = self._filter_aligned_prefix(
            remaining_correct_solution_steps,
            aligned_facts,
        )
        if not accepted_facts:
            return self._update_last_step_record(
                state=state,
                current_correct_state=list(state.get("current_correct_state", [])),
            )

        updated_correct_state = self._merge_facts(
            state.get("current_correct_state", []),
            accepted_facts,
        )
        return {
            "current_correct_state": updated_correct_state,
            "correct_solution_cursor": state.get("correct_solution_cursor", 0) + len(accepted_facts),
            **self._update_last_step_record(
                state=state,
                current_correct_state=updated_correct_state,
            ),
        }

    def advance_step(self, state: WorkflowState) -> dict[str, Any]:
        return {
            "current_step_index": state["current_step_index"] + 1,
            "current_step_text": "",
            "step_intent": "",
            "math_tool_request": None,
            "math_tool_result": None,
            "current_step_status": "correct",
            "current_step_judge_reason": "",
        }

    def check_completion(self, state: WorkflowState) -> dict[str, Any]:
        system_prompt, user_prompt = build_completion_prompt(
            state.get("shared_stem_text", ""),
            state.get("shared_problem_context", []),
            state["current_subproblem_text"],
            state.get("current_student_steps", []),
            state.get("current_student_state", []),
        )
        result = self.llm.invoke_structured(CompletionCheckResult, system_prompt, user_prompt)
        if result.is_complete:
            return {"completion_is_complete": True}

        student_state_before = list(state.get("current_student_state", []))
        correct_state_before = list(state.get("current_correct_state", []))
        step_record = StepRecord(
            step_index=len(state.get("current_student_steps", [])) + 1,
            step_text="学生未完成当前子问题",
            step_intent="作答未完成",
            step_status="autonomous_error",
            judge_reason=result.issue_reason,
            student_state_before=student_state_before,
            correct_state_before=correct_state_before,
            student_state_after=student_state_before,
            correct_state_after=correct_state_before,
        )
        updated_step_records = [*state.get("step_records", []), step_record.model_dump()]

        return {
            "completion_is_complete": False,
            "current_step_text": step_record.step_text,
            "step_intent": step_record.step_intent,
            "current_step_status": step_record.step_status,
            "current_step_judge_reason": result.issue_reason,
            "step_records": updated_step_records,
            "history_has_error": True,
            "current_is_correct": False,
        }

    def finalize_subproblem(self, state: WorkflowState) -> dict[str, Any]:
        result = SubproblemResult(
            subproblem_index=state["current_subproblem_index"] + 1,
            is_correct=state.get("current_is_correct", True),
            student_steps=state.get("current_student_steps", []),
            step_records=[StepRecord(**item) for item in state.get("step_records", [])],
            correct_solution_steps=state.get("correct_solution_steps", []),
            error_reports=[],
        )
        updated_results = deepcopy(state.get("subproblem_results", []))
        updated_results.append(result.model_dump())
        return {
            "subproblem_results": updated_results,
            "current_subproblem_index": state["current_subproblem_index"] + 1,
        }

    def summarize_grading(self, state: WorkflowState) -> dict[str, Any]:
        return {"is_correct": all(item["is_correct"] for item in state.get("subproblem_results", []))}

    def attribute_errors(self, state: WorkflowState) -> dict[str, Any]:
        updated_results = deepcopy(state.get("subproblem_results", []))

        for index, item in enumerate(updated_results):
            error_reports: list[dict[str, Any]] = []
            step_records = item.get("step_records", [])

            for step_position, step_record in enumerate(step_records):
                if step_record.get("step_status") != "autonomous_error":
                    continue

                correct_next_step = self._get_correct_next_step(
                    step_record.get("correct_state_before", []),
                    item.get("correct_solution_steps", []),
                )
                system_prompt, user_prompt = build_attribution_prompt(
                    shared_stem_text=state.get("shared_stem_text", ""),
                    shared_problem_context=state.get("shared_problem_context", []),
                    current_subproblem_text=state["subproblems"][item["subproblem_index"] - 1],
                    step_record=step_record,
                    correct_next_step=correct_next_step,
                )
                attribution = self.llm.invoke_structured(
                    AttributionResult,
                    system_prompt,
                    user_prompt,
                )
                primary_reason_type = (attribution.primary_reason_type or "").strip()
                reason_type = [item.strip() for item in attribution.reason_type if item and item.strip()]
                if primary_reason_type and primary_reason_type not in reason_type:
                    reason_type = [primary_reason_type, *reason_type]
                error_report = ErrorReport(
                    step_index=step_record["step_index"],
                    step_text=step_record.get("step_text", ""),
                    judge_reason=step_record.get("judge_reason", ""),
                    primary_reason_type=primary_reason_type,
                    reason_type=reason_type,
                    attribution_reason=attribution.attribution_reason,
                    feedback_text=attribution.feedback_text,
                )
                error_reports.append(error_report.model_dump())

            updated_results[index]["error_reports"] = error_reports

        return {"subproblem_results": updated_results}

    def finalize_output(self, state: WorkflowState) -> dict[str, Any]:
        return {
            "is_correct": state["is_correct"],
            "subproblem_results": state["subproblem_results"],
        }

    def _build_review_source_text(self, state: WorkflowState) -> str:
        shared_stem_text = (state.get("shared_stem_text") or "").strip()
        if shared_stem_text:
            return shared_stem_text
        return state["question_text"]

    def _merge_facts(self, base: list[str], updates: list[str]) -> list[str]:
        merged = list(base)
        for fact in self._normalize_facts(updates):
            if fact not in merged:
                merged.append(fact)
        return merged

    def _normalize_facts(self, facts: list[str] | None) -> list[str]:
        normalized: list[str] = []
        for fact in facts or []:
            text = fact.strip()
            if text:
                normalized.append(text)
        return normalized

    def _filter_aligned_prefix(
        self,
        remaining_correct_solution_steps: list[str],
        aligned_facts: list[str],
    ) -> list[str]:
        accepted: list[str] = []
        for expected, actual in zip(remaining_correct_solution_steps, aligned_facts):
            if expected != actual:
                break
            accepted.append(actual)
        return accepted

    def _get_correct_next_step(
        self,
        correct_state_before: list[str],
        correct_solution_steps: list[str],
    ) -> str:
        matched_prefix_count = 0
        for fact in correct_solution_steps:
            if fact in correct_state_before:
                matched_prefix_count += 1
                continue
            break
        if matched_prefix_count < len(correct_solution_steps):
            return correct_solution_steps[matched_prefix_count]
        return ""

    def _update_last_step_record(
        self,
        state: WorkflowState,
        current_correct_state: list[str],
    ) -> dict[str, Any]:
        step_records = deepcopy(state.get("step_records", []))
        if step_records:
            step_records[-1]["correct_state_after"] = list(current_correct_state)
        return {"step_records": step_records}
