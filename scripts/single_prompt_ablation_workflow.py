from __future__ import annotations

import json
import os
import time
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.5-plus")
QWEN_TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", "0.5"))

ERROR_LABELS = [
    "公式与法则误用",
    "基础计算失误",
    "审题遗漏与条件忽视",
    "概念混淆与理解偏差",
    "逻辑推断失误",
    "其他未明确错误",
]


class ProblemParseBlock(BaseModel):
    is_multi_part: bool
    shared_stem_text: str = Field(default="")
    subproblems: list[str] = Field(default_factory=list)
    student_subanswers: list[str] = Field(default_factory=list)


class StepEvaluation(BaseModel):
    step_index: int
    step_text: str = Field(default="")
    step_intent: str = Field(default="")
    step_status: Literal["correct", "autonomous_error", "inherited_error"] = "correct"
    judge_reason: str = Field(default="")


class ErrorReportBlock(BaseModel):
    step_index: int
    step_text: str = Field(default="")
    judge_reason: str = Field(default="")
    primary_reason_type: str = Field(default="")
    reason_type: list[str] = Field(default_factory=list)
    attribution_reason: str = Field(default="")
    feedback_text: str = Field(default="")


class SubproblemEvaluation(BaseModel):
    subproblem_index: int
    subproblem_text: str = Field(default="")
    student_answer_segment: str = Field(default="")
    student_steps: list[str] = Field(default_factory=list)
    step_evaluations: list[StepEvaluation] = Field(default_factory=list)
    error_reports: list[ErrorReportBlock] = Field(default_factory=list)
    is_correct: bool


class FinalPredictionBlock(BaseModel):
    is_correct: bool
    incorrect_subproblem_indices: list[int] = Field(default_factory=list)
    autonomous_error_count: int = 0
    overall_feedback_text: str = Field(default="")


class SinglePromptAblationOutput(BaseModel):
    problem_parse: ProblemParseBlock
    subproblem_evaluations: list[SubproblemEvaluation] = Field(default_factory=list)
    final_prediction: FinalPredictionBlock


def _section(title: str, content: str) -> str:
    normalized_content = content.strip() if content.strip() else "（空）"
    return f"【{title}】\n{normalized_content}\n"


def build_single_prompt_messages(question_text: str, student_answer_text: str) -> tuple[str, str]:
    error_labels_text = "\n".join(f"- {label}" for label in ERROR_LABELS)
    system_prompt = (
        "你是一位经验丰富的数学教师，擅长对解答题进行分步批改与错误归因。\n"
        "你的批改必须严格依据题目原文与学生作答原文，不得引入外部假设，不得替学生补写其未写出的新推导。\n"
        "你需要在单次输出中完成完整批改，但必须保持“先过程，后结论”的顺序。\n"
        "顶层字段必须按以下顺序输出，且不得交换顺序：\n"
        "1. problem_parse\n"
        "2. subproblem_evaluations\n"
        "3. final_prediction\n\n"
        "总原则：\n"
        "1. problem_parse 只负责识别题目结构、公共题干和作答对齐，不提前给最终结论。\n"
        "2. subproblem_evaluations 必须按子问题顺序展开；每个子问题内部必须先给出 student_steps，再给出 step_evaluations，再给出 error_reports，最后给出该子问题 is_correct。\n"
        "3. student_steps、step_text、student_answer_segment 应尽量保留学生原始作答中的文字、符号与公式，不要改写原意，不要补写学生未写出的新中间推导。只有占位步骤“学生未完成当前子问题”允许额外生成。\n"
        "4. step_evaluations 必须与 student_steps 严格一一对应，长度一致，step_index 从 1 开始连续编号。\n"
        "5. step_status 只允许使用 correct、autonomous_error、inherited_error。\n"
        "6. correct：当前步在客观数学上成立，并且不依赖先前错误前提。\n"
        "7. autonomous_error：当前步本身存在独立错误；或学生在尚未完成目标时中途终止作答。\n"
        "8. inherited_error：当前步推导动作本身可以顺着前面的错误继续进行，但由于依赖了同一子问题中更早出现的错误前提，结果偏离正确路径。只有在同一子问题前面已出现 autonomous_error 时，后续步骤才允许标为 inherited_error。\n"
        "9. 若学生某个子问题作答中途停止、未形成完整结论，必须在 student_steps 末尾补一个占位步骤“学生未完成当前子问题”，并为它生成对应的 step_evaluation，且该步 step_status 必须为 autonomous_error。\n"
        "10. 若学生某一步出现明显笔误、漏写或短暂错误，但在后续步骤中已明确自我修正，且修正后推导链条完整正确，则应优先按修正后的有效解题过程切步与判定，不必机械保留一个已被修正的 autonomous_error。\n"
        "11. 每个子问题中的 error_reports 只能覆盖该子问题中的 autonomous_error，且与 step_evaluations 中对应的 autonomous_error 步骤一一对应；同一子问题可以有零条、一条或多条 error_reports。\n"
        "12. 不要为 inherited_error 单独生成 error_report；它们只是沿用前面错误后的后续步骤。\n"
        "13. 每条 error_report 中的 step_index、step_text、judge_reason 必须与对应 autonomous_error 步骤保持一致。\n"
        "14. primary_reason_type 必须从下列标签中选择一个；reason_type 可以包含一个或多个标签，但必须包含 primary_reason_type，且不得重复：\n"
        f"{error_labels_text}\n"
        "15. 每条 error_report 必须包含 step_index、step_text、judge_reason、primary_reason_type、reason_type、attribution_reason、feedback_text。\n"
        "16. attribution_reason 要用 2-3 句话分析错误发生的根本原因，聚焦学生的思维偏差，而不是只重复表面现象。\n"
        "17. feedback_text 必须直接面向学生，先指出关键问题，再给出修正方向；若整体正确，可给出简短、具体的正向反馈。\n"
        "18. 子问题 is_correct 只有在该子问题不存在未修正的 autonomous_error 或 inherited_error，且最终结论完整正确时，才能为 true。\n"
        "19. final_prediction 只能总结前面字段中已经出现的信息，不能新增新的证据、步骤、标签或结论。\n"
        "20. final_prediction 只输出整体结论与摘要：is_correct、incorrect_subproblem_indices、autonomous_error_count、overall_feedback_text。\n"
        "21. incorrect_subproblem_indices 必须恰好等于所有 is_correct=false 的子问题编号列表。\n"
        "22. autonomous_error_count 必须恰好等于所有子问题中 error_reports 的总数。\n"
        "23. 只有当所有子问题 is_correct 都为 true 时，final_prediction.is_correct 才能为 true。\n"
        "24. judge_reason、attribution_reason、feedback_text、overall_feedback_text 都应具体、简洁，避免“步骤正确”“步骤错误”这类空洞表述。\n"
    )
    user_prompt = (
        "请对以下数学解答题及学生作答进行详细批改。\n\n"
        "【一、题目与作答结构识别】\n"
        "首先，请判断本题是否为多问结构。若包含多个小问，请提取公共题干，并将各小问与学生作答段落一一对齐；若为单问，则公共题干置空，直接将完整题干与完整作答作为单独一项处理。\n\n"
        "【二、逐子问题分步批改】\n"
        "对于每一个子问题，请按以下层次展开：\n"
        "1. 将学生在该子问题下的作答切分为若干有意义的步骤。若作答中途终止、未得出最终结论，请在末尾补充一个占位步骤“学生未完成当前子问题”。\n"
        "2. 对每一步进行评价，包含：步骤序号与原文、步骤意图、步骤状态、判定理由。\n"
        "3. 若该子问题中存在 autonomous_error 步骤，请为每一个此类步骤生成一条错误报告。\n"
        "4. 最后给出该子问题的整体正误判定 is_correct。\n\n"
        "【三、错误类型标签】\n"
        "请从以下标签中选择：\n"
        f"{error_labels_text}\n"
        "若确实无法归入上述任何一类，可使用“其他未明确错误”。\n\n"
        "【四、整体结论】\n"
        "在所有子问题批改完成后，请汇总给出：整题是否全对（is_correct）、出错的子问题序号列表（incorrect_subproblem_indices）、自主错误总次数（autonomous_error_count）、整体反馈（overall_feedback_text）。\n\n"
        "【五、额外提醒】\n"
        "- 若为单题：shared_stem_text 置空，subproblems 仅保留完整题干，student_subanswers 仅保留完整作答。\n"
        "- 若为多题：shared_stem_text 保留公共题干，subproblems 仅保留各小问自身，student_subanswers 与 subproblems 严格对齐。\n"
        "- step_intent 用一句话概括学生当前步想做什么。\n"
        "- 每个子问题里，如果识别出多个自主错误，应输出多条 error_reports，并按步骤先后排序。\n"
        "- 若学生最终答案正确，但中间错误未被修正，则整体仍判错。\n"
        "- 输出格式必须为严格 JSON；字段顺序请遵循本节描述逻辑，具体字段名以最后给出的 JSON 示例为准。\n\n"
        f"{_section('题目原文', question_text)}\n"
        f"{_section('学生作答原文', student_answer_text)}\n"
        "请直接输出如下结构的 JSON 对象，不要包含任何注释、解释或额外说明：\n"
        "{\n"
        '  "problem_parse": {\n'
        '    "subproblems": ["子问题1题干", "子问题2题干"],\n'
        '    "student_subanswers": ["子问题1作答", "子问题2作答"],\n'
        '    "shared_stem_text": "公共题干（若无则为空字符串）",\n'
        '    "is_multi_part": true\n'
        "  },\n"
        '  "subproblem_evaluations": [\n'
        "    {\n"
        '      "subproblem_index": 1,\n'
        '      "subproblem_text": "子问题1题干",\n'
        '      "student_answer_segment": "子问题1对应作答",\n'
        '      "student_steps": ["步骤1原文", "步骤2原文"],\n'
        '      "step_evaluations": [\n'
        "        {\n"
        '          "step_index": 1,\n'
        '          "step_text": "步骤原文",\n'
        '          "step_intent": "本步意图",\n'
        '          "step_status": "correct",\n'
        '          "judge_reason": "判定理由"\n'
        "        }\n"
        "      ],\n"
        '      "error_reports": [\n'
        "        {\n"
        '          "step_index": 2,\n'
        '          "step_text": "出错步骤原文",\n'
        '          "judge_reason": "判定理由",\n'
        '          "primary_reason_type": "主要错误类型",\n'
        '          "reason_type": ["类型1", "类型2"],\n'
        '          "attribution_reason": "归因分析文本",\n'
        '          "feedback_text": "面向学生的反馈建议"\n'
        "        }\n"
        "      ],\n"
        '      "is_correct": false\n'
        "    }\n"
        "  ],\n"
        '  "final_prediction": {\n'
        '    "is_correct": false,\n'
        '    "incorrect_subproblem_indices": [1],\n'
        '    "autonomous_error_count": 1,\n'
        '    "overall_feedback_text": "整体反馈"\n'
        "  }\n"
        "}"
    )
    return system_prompt, user_prompt


def create_llm_client() -> ChatOpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未检测到 DASHSCOPE_API_KEY，请先在系统环境变量中配置它。")
    if not QWEN_BASE_URL:
        raise ValueError("未检测到 QWEN_BASE_URL，请先在系统环境变量中配置它。")

    return ChatOpenAI(
        api_key=api_key,
        base_url=QWEN_BASE_URL,
        model=QWEN_MODEL,
        temperature=QWEN_TEMPERATURE,
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
                continue
            if "text" in item:
                parts.append(str(item["text"]))
                continue
            if "content" in item:
                parts.append(str(item["content"]))
        return "\n".join(part.strip() for part in parts if str(part).strip()).strip()
    return str(content).strip()


def _strip_code_fences(text: str) -> str:
    normalized = text.strip()
    if not normalized.startswith("```"):
        return normalized

    lines = normalized.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _to_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _extract_json_payload(raw_text: str) -> Any:
    candidates = []
    stripped = raw_text.strip()
    if stripped:
        candidates.append(stripped)
        stripped_fences = _strip_code_fences(stripped)
        if stripped_fences != stripped:
            candidates.append(stripped_fences)

    decoder = json.JSONDecoder()
    last_error: Exception | None = None

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc

        start_index = candidate.find("{")
        if start_index == -1:
            continue
        try:
            payload, _ = decoder.raw_decode(candidate[start_index:])
            return payload
        except json.JSONDecodeError as exc:
            last_error = exc

    raise ValueError("模型输出中未找到可解析的 JSON 对象。") from last_error


def _normalize_text(value: str | None) -> str:
    return value.strip() if value else ""


def _normalize_text_list(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        text = value.strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_reason_types(primary_reason_type: str, reason_types: list[str] | None) -> list[str]:
    normalized = _normalize_text_list(reason_types)
    if primary_reason_type and primary_reason_type not in normalized:
        normalized.insert(0, primary_reason_type)
    return normalized


def parse_single_prompt_output(raw_text: str) -> SinglePromptAblationOutput:
    payload = _extract_json_payload(raw_text)
    return SinglePromptAblationOutput.model_validate(payload)


def _collect_autonomous_error_steps(
    output: SinglePromptAblationOutput,
) -> list[tuple[int, str, StepEvaluation]]:
    error_steps: list[tuple[int, str, StepEvaluation]] = []
    for subproblem in output.subproblem_evaluations:
        subproblem_text = _normalize_text(subproblem.subproblem_text)
        for step in subproblem.step_evaluations:
            if step.step_status == "autonomous_error":
                error_steps.append((subproblem.subproblem_index, subproblem_text, step))
    return error_steps


def _collect_error_report_map(
    output: SinglePromptAblationOutput,
) -> dict[tuple[int, int], ErrorReportBlock]:
    report_map: dict[tuple[int, int], ErrorReportBlock] = {}
    for subproblem in output.subproblem_evaluations:
        for report in subproblem.error_reports:
            report_map[(subproblem.subproblem_index, report.step_index)] = report
    return report_map


def build_prediction(output: SinglePromptAblationOutput) -> dict[str, Any]:
    autonomous_error_steps = _collect_autonomous_error_steps(output)
    report_map = _collect_error_report_map(output)
    has_non_correct_step = any(
        step.step_status != "correct"
        for subproblem in output.subproblem_evaluations
        for step in subproblem.step_evaluations
    )

    process_is_correct = (
        not has_non_correct_step
        and all(subproblem.is_correct for subproblem in output.subproblem_evaluations)
    )

    flattened_error_reports: list[dict[str, Any]] = []
    for subproblem_index, subproblem_text, step in autonomous_error_steps:
        report = report_map.get((subproblem_index, step.step_index))
        primary_reason_type = _normalize_text(report.primary_reason_type if report else "")
        flattened_error_reports.append(
            {
                "subproblem_index": subproblem_index,
                "subproblem_text": subproblem_text,
                "step_index": step.step_index,
                "step_text": _normalize_text(step.step_text),
                "step_intent": _normalize_text(step.step_intent),
                "judge_reason": _normalize_text(
                    report.judge_reason if report and report.judge_reason else step.judge_reason
                ),
                "primary_reason_type": primary_reason_type,
                "reason_type": _normalize_reason_types(
                    primary_reason_type,
                    report.reason_type if report else [],
                ),
                "attribution_reason": _normalize_text(report.attribution_reason if report else ""),
                "feedback_text": _normalize_text(report.feedback_text if report else ""),
            }
        )

    incorrect_subproblem_indices = [
        subproblem.subproblem_index
        for subproblem in output.subproblem_evaluations
        if not subproblem.is_correct
    ]

    overall_feedback_text = _normalize_text(output.final_prediction.overall_feedback_text)
    if process_is_correct and not overall_feedback_text:
        overall_feedback_text = "解答整体正确，关键推导链条基本完整。"
    if not process_is_correct and not overall_feedback_text and flattened_error_reports:
        overall_feedback_text = flattened_error_reports[0]["feedback_text"]

    return {
        "is_correct": process_is_correct,
        "incorrect_subproblem_indices": incorrect_subproblem_indices,
        "autonomous_error_count": len(flattened_error_reports),
        "overall_feedback_text": overall_feedback_text,
        "error_reports": flattened_error_reports,
    }


def _extract_token_usage(payload: dict[str, Any], usage_metadata: dict[str, Any]) -> dict[str, Any]:
    input_tokens = usage_metadata.get("input_tokens")
    output_tokens = usage_metadata.get("output_tokens")
    total_tokens = usage_metadata.get("total_tokens")

    if input_tokens is None:
        input_tokens = payload.get("prompt_tokens")
    if output_tokens is None:
        output_tokens = payload.get("completion_tokens")
    if total_tokens is None:
        total_tokens = payload.get("total_tokens")

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "raw_token_usage": payload,
    }


def call_single_prompt_model(question_text: str, student_answer_text: str) -> dict[str, Any]:
    client = create_llm_client()
    system_prompt, user_prompt = build_single_prompt_messages(question_text, student_answer_text)

    started_at = time.perf_counter()
    ai_message = client.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    duration_seconds = round(time.perf_counter() - started_at, 6)

    response_metadata = _to_jsonable(getattr(ai_message, "response_metadata", {}) or {})
    usage_metadata = _to_jsonable(getattr(ai_message, "usage_metadata", {}) or {})
    token_usage_payload = response_metadata.get("token_usage", {}) or {}

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_text": _content_to_text(ai_message.content),
        "timing": {
            "model_call_duration_seconds": duration_seconds,
        },
        "finish_reason": response_metadata.get("finish_reason"),
        "response_metadata": response_metadata,
        "usage_metadata": usage_metadata,
        "tokens": _extract_token_usage(token_usage_payload, usage_metadata),
    }
