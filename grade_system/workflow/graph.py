from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from grade_system.models.state import WorkflowState
from grade_system.services.llm_service import QwenWorkflowLLM
from grade_system.tools.math_tools import MathToolEngine
from grade_system.workflow.nodes import WorkflowNodes


def build_grading_graph(
    llm: QwenWorkflowLLM | None = None,
    math_tool: MathToolEngine | None = None,
):
    nodes = WorkflowNodes(llm or QwenWorkflowLLM(), math_tool or MathToolEngine())
    builder = StateGraph(WorkflowState)

    builder.add_node("parse_problem", nodes.parse_problem)
    builder.add_node("review_problem", nodes.review_problem)
    builder.add_node("prepare_subproblem", nodes.prepare_subproblem)
    builder.add_node("split_steps", nodes.split_steps)
    builder.add_node("analyze_step", nodes.analyze_step)
    builder.add_node("run_math_tool", nodes.run_math_tool)
    builder.add_node("judge_step", nodes.judge_step)
    builder.add_node("generate_correct_solution", nodes.generate_correct_solution)
    builder.add_node("align_correct_frontier", nodes.align_correct_frontier)
    builder.add_node("advance_step", nodes.advance_step)
    builder.add_node("check_completion", nodes.check_completion)
    builder.add_node("finalize_subproblem", nodes.finalize_subproblem)
    builder.add_node("summarize_grading", nodes.summarize_grading)
    builder.add_node("attribute_errors", nodes.attribute_errors)
    builder.add_node("finalize_output", nodes.finalize_output)

    builder.add_edge(START, "parse_problem")
    builder.add_edge("parse_problem", "review_problem")
    builder.add_edge("review_problem", "prepare_subproblem")
    builder.add_edge("prepare_subproblem", "split_steps")

    builder.add_conditional_edges(
        "split_steps",
        route_after_step_split,
        {"analyze_step": "analyze_step", "check_completion": "check_completion"},
    )
    builder.add_conditional_edges(
        "analyze_step",
        route_after_step_analysis,
        {"run_math_tool": "run_math_tool", "judge_step": "judge_step"},
    )
    builder.add_edge("run_math_tool", "judge_step")
    builder.add_conditional_edges(
        "judge_step",
        route_after_step_judgment,
        {
            "generate_correct_solution": "generate_correct_solution",
            "align_correct_frontier": "align_correct_frontier",
            "advance_step": "advance_step",
        },
    )
    builder.add_conditional_edges(
        "generate_correct_solution",
        route_after_correct_solution,
        {"advance_step": "advance_step", "finalize_subproblem": "finalize_subproblem"},
    )
    builder.add_edge("align_correct_frontier", "advance_step")
    builder.add_conditional_edges(
        "advance_step",
        route_after_advance_step,
        {"analyze_step": "analyze_step", "check_completion": "check_completion"},
    )
    builder.add_conditional_edges(
        "check_completion",
        route_after_completion_check,
        {
            "generate_correct_solution": "generate_correct_solution",
            "finalize_subproblem": "finalize_subproblem",
        },
    )
    builder.add_conditional_edges(
        "finalize_subproblem",
        route_after_subproblem,
        {"prepare_subproblem": "prepare_subproblem", "summarize_grading": "summarize_grading"},
    )
    builder.add_edge("summarize_grading", "attribute_errors")
    builder.add_edge("attribute_errors", "finalize_output")
    builder.add_edge("finalize_output", END)

    return builder.compile()


def route_after_step_split(state: WorkflowState) -> Literal["analyze_step", "check_completion"]:
    return "analyze_step" if state.get("current_student_steps") else "check_completion"


def route_after_step_analysis(state: WorkflowState) -> Literal["run_math_tool", "judge_step"]:
    return "run_math_tool" if state.get("math_tool_request") else "judge_step"


def route_after_step_judgment(
    state: WorkflowState,
) -> Literal["generate_correct_solution", "align_correct_frontier", "advance_step"]:
    if state.get("current_step_status") != "correct" and not state.get("correct_solution_started", False):
        return "generate_correct_solution"
    if state.get("correct_solution_started", False):
        return "align_correct_frontier"
    return "advance_step"


def route_after_correct_solution(
    state: WorkflowState,
) -> Literal["advance_step", "finalize_subproblem"]:
    if state.get("current_step_index", 0) < len(state.get("current_student_steps", [])):
        return "advance_step"
    return "finalize_subproblem"


def route_after_advance_step(state: WorkflowState) -> Literal["analyze_step", "check_completion"]:
    if state.get("current_step_index", 0) < len(state.get("current_student_steps", [])):
        return "analyze_step"
    return "check_completion"


def route_after_completion_check(
    state: WorkflowState,
) -> Literal["generate_correct_solution", "finalize_subproblem"]:
    if state.get("completion_is_complete", True):
        return "finalize_subproblem"
    if not state.get("correct_solution_started", False):
        return "generate_correct_solution"
    return "finalize_subproblem"


def route_after_subproblem(
    state: WorkflowState,
) -> Literal["prepare_subproblem", "summarize_grading"]:
    if state["current_subproblem_index"] < len(state["subproblems"]):
        return "prepare_subproblem"
    return "summarize_grading"


def extract_final_output(state: WorkflowState) -> dict:
    return {
        "is_correct": state["is_correct"],
        "subproblem_results": state["subproblem_results"],
    }
