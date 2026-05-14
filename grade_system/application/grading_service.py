from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from grade_system.config import Settings, load_settings
from grade_system.models.schemas import GradeWorkflowInput, GradeWorkflowOutput
from grade_system.persistence.models import GradingTask
from grade_system.persistence.repositories.grading_task_repository import (
    GradingTaskRepository,
)
from grade_system.services.llm_service import QwenWorkflowLLM
from grade_system.workflow.graph import build_grading_graph, extract_final_output

SUCCESS_SUMMARY_TEXT = "答案正确，无需分析错因"
DEFAULT_FAILURE_SUMMARY_TEXT = "暂未生成详细错因，请结合批改结果继续检查。"
DEFAULT_SYSTEM_ERROR_TEXT = "批改服务暂时不可用，请稍后重试。"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def build_question_excerpt(question_text: str, max_length: int = 60) -> str:
    normalized = " ".join(question_text.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3]}..."


def build_attribution_summary(final_output: GradeWorkflowOutput) -> str:
    if final_output.is_correct:
        return SUCCESS_SUMMARY_TEXT

    snippets: list[str] = []
    for subproblem in final_output.subproblem_results:
        for error_report in subproblem.error_reports:
            for candidate in (
                error_report.feedback_text,
                error_report.attribution_reason,
                error_report.judge_reason,
            ):
                text = candidate.strip()
                if text and text not in snippets:
                    snippets.append(text)
                    break

    if not snippets:
        for subproblem in final_output.subproblem_results:
            for step_record in subproblem.step_records:
                if step_record.step_status == "correct":
                    continue
                text = step_record.judge_reason.strip()
                if text and text not in snippets:
                    snippets.append(text)

    if not snippets:
        return DEFAULT_FAILURE_SUMMARY_TEXT

    return "\n".join(snippets[:3])


def normalize_workflow_error_message(exc: Exception) -> str:
    raw_message = str(exc).strip()
    lowered_message = raw_message.lower()
    exception_name = type(exc).__name__.lower()

    if "timeout" in lowered_message or "timed out" in lowered_message or "timeout" in exception_name:
        return "模型服务响应超时，当前题目已停止批改，请稍后重试。"
    if "connection error" in lowered_message or "connect" in lowered_message or "apiconnection" in exception_name:
        return "模型服务连接失败，当前题目已停止批改，请稍后重试。"
    if raw_message:
        return raw_message
    return DEFAULT_SYSTEM_ERROR_TEXT


@dataclass
class WorkflowExecutionResult:
    final_output: GradeWorkflowOutput
    usage: dict[str, int]
    attribution_summary_text: str


class GradingWorkflowService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()

    def execute(self, payload: GradeWorkflowInput) -> WorkflowExecutionResult:
        llm = QwenWorkflowLLM(self.settings)
        llm.reset_usage_totals()
        graph = build_grading_graph(llm=llm)
        final_state = graph.invoke(payload.model_dump())
        final_output = GradeWorkflowOutput.model_validate(extract_final_output(final_state))
        return WorkflowExecutionResult(
            final_output=final_output,
            usage=llm.get_usage_totals(),
            attribution_summary_text=build_attribution_summary(final_output),
        )


class GradingTaskService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.workflow_service = GradingWorkflowService(self.settings)

    def create_task(
        self,
        session: Session,
        *,
        user_id: int,
        question_text: str,
        student_answer_text: str,
    ) -> GradingTask:
        repository = GradingTaskRepository(session)
        task = repository.create(
            user_id=user_id,
            question_text=question_text,
            student_answer_text=student_answer_text,
            question_excerpt=build_question_excerpt(question_text),
            request_snapshot_json={
                "question_text": question_text,
                "student_answer_text": student_answer_text,
            },
            workflow_version=self.settings.grading_workflow_version,
        )
        session.flush()
        return task

    def run_task(self, session: Session, task: GradingTask) -> GradingTask:
        repository = GradingTaskRepository(session)
        repository.mark_running(task)
        session.commit()

        try:
            payload = GradeWorkflowInput(
                question_text=task.question_text,
                student_answer_text=task.student_answer_text,
            )
            execution_result = self.workflow_service.execute(payload)
            repository.mark_succeeded(
                task,
                final_output=execution_result.final_output.model_dump(mode="json"),
                is_correct=execution_result.final_output.is_correct,
                attribution_summary_text=execution_result.attribution_summary_text,
                usage=execution_result.usage,
            )
            session.commit()
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            task = repository.get_by_task_id(task.task_id)
            if task is None:
                raise
            repository.mark_failed(task, normalize_workflow_error_message(exc))
            session.commit()
        return task
