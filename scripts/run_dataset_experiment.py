from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grade_system.models.schemas import GradeWorkflowOutput
from grade_system.workflow.graph import build_grading_graph, extract_final_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the grading workflow on a JSONL dataset with sliding-window concurrency."
    )
    parser.add_argument(
        "--input",
        default=str(ROOT / "DataSet" / "pilot_testset_30.jsonl"),
        help="Input JSONL dataset path.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "ExperimentResults" / "pilot_testset_30_results.jsonl"),
        help="Output JSONL results path.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent workflow runs.",
    )
    return parser.parse_args()


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_index, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record["_dataset_index"] = line_index
            records.append(record)
    return records


def extract_first_error_prediction(workflow_output: dict[str, Any]) -> dict[str, Any]:
    first_error_report: dict[str, Any] | None = None
    first_subproblem_index: int | None = None
    first_step_index: int | None = None
    total_error_report_count = 0

    for subproblem in workflow_output.get("subproblem_results", []):
        subproblem_index = subproblem.get("subproblem_index")
        for error_report in subproblem.get("error_reports", []):
            total_error_report_count += 1
            step_index = error_report.get("step_index")
            if first_error_report is None:
                first_error_report = error_report
                first_subproblem_index = subproblem_index
                first_step_index = step_index
                continue
            if (subproblem_index, step_index) < (first_subproblem_index, first_step_index):
                first_error_report = error_report
                first_subproblem_index = subproblem_index
                first_step_index = step_index

    if first_error_report is None:
        return {
            "first_error_subproblem_index": None,
            "first_error_step_index": None,
            "primary_reason_type": "",
            "reason_type": [],
            "error_report_count": total_error_report_count,
        }

    reason_type = first_error_report.get("reason_type") or []
    normalized_reason_type = [item.strip() for item in reason_type if item and item.strip()]
    return {
        "first_error_subproblem_index": first_subproblem_index,
        "first_error_step_index": first_step_index,
        "primary_reason_type": (first_error_report.get("primary_reason_type") or "").strip(),
        "reason_type": normalized_reason_type,
        "error_report_count": total_error_report_count,
    }


def run_single_record(record: dict[str, Any]) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    workflow_input = {
        "question_text": record["question"],
        "student_answer_text": record["answer"],
    }

    try:
        graph = build_grading_graph()
        final_state = graph.invoke(workflow_input)
        workflow_output = GradeWorkflowOutput.model_validate(
            extract_final_output(final_state)
        ).model_dump()
        prediction = {
            "is_correct": workflow_output.get("is_correct"),
            **extract_first_error_prediction(workflow_output),
        }
        status = "success"
        error_payload = None
    except Exception as exc:  # pragma: no cover - runtime protection
        workflow_output = None
        prediction = {
            "is_correct": None,
            "first_error_subproblem_index": None,
            "first_error_step_index": None,
            "primary_reason_type": "",
            "reason_type": [],
            "error_report_count": 0,
        }
        status = "error"
        error_payload = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    finished_at = datetime.now(timezone.utc)
    duration_seconds = round(time.perf_counter() - start_time, 6)

    return {
        "dataset_index": record["_dataset_index"],
        "sample_id": record.get("id"),
        "reference_id": record.get("reference_id"),
        "subject": record.get("subject"),
        "level": record.get("level"),
        "status": status,
        "gold": {
            "is_correct": record.get("is_correct"),
            "error_category": record.get("error_category", ""),
        },
        "workflow_input": workflow_input,
        "prediction": prediction,
        "timing": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration_seconds,
        },
        "workflow_output": workflow_output,
        "error": error_payload,
    }


def write_results(
    records: list[dict[str, Any]],
    output_path: Path,
    max_workers: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(records)
    submitted = 0
    run_start_time = time.perf_counter()

    print(
        (
            f"Starting experiment run: total_samples={total}, "
            f"max_workers={max_workers}, output={output_path}"
        ),
        flush=True,
    )

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            record_iter = iter(records)
            pending: dict[Future[dict[str, Any]], dict[str, Any]] = {}
            completed = 0

            def submit_next() -> bool:
                nonlocal submitted
                try:
                    next_record = next(record_iter)
                except StopIteration:
                    return False
                future = executor.submit(run_single_record, next_record)
                pending[future] = next_record
                submitted += 1
                print(
                    (
                        f"Submitted sample {submitted}/{total}: "
                        f"id={next_record.get('id')} ref={next_record.get('reference_id')}"
                    ),
                    flush=True,
                )
                return True

            initial_window = min(max_workers, total)
            for _ in range(initial_window):
                submit_next()

            while pending:
                done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for future in done:
                    source_record = pending.pop(future)
                    result = future.result()
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()
                    completed += 1
                    print(
                        (
                            f"[{completed}/{total}] "
                            f"id={source_record.get('id')} "
                            f"status={result['status']} "
                            f"duration={result['timing']['duration_seconds']:.3f}s"
                        ),
                        flush=True,
                    )
                    submit_next()

    total_wall_time = time.perf_counter() - run_start_time
    print(
        f"Experiment run finished. results_written_to={output_path} total_wall_time_seconds={total_wall_time:.3f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    records = load_jsonl_records(input_path)
    if not records:
        raise ValueError(f"No records found in dataset: {input_path}")

    write_results(records, output_path, args.max_workers)


if __name__ == "__main__":
    main()
