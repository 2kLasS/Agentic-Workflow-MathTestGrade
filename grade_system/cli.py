from __future__ import annotations

import argparse
import json
from pathlib import Path

from grade_system.models.schemas import GradeWorkflowInput, GradeWorkflowOutput
from grade_system.workflow.graph import build_grading_graph, extract_final_output


def main() -> None:
    parser = argparse.ArgumentParser(description="数学解答题批改与错误归因工作流")
    parser.add_argument("--input", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--pretty", action="store_true", help="格式化输出 JSON")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    workflow_input = GradeWorkflowInput.model_validate(payload)

    graph = build_grading_graph()
    final_state = graph.invoke(workflow_input.model_dump())
    final_output = GradeWorkflowOutput.model_validate(extract_final_output(final_state))

    print(
        json.dumps(
            final_output.model_dump(),
            ensure_ascii=False,
            indent=2 if args.pretty else None,
        )
    )


