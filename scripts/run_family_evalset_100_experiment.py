from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grade_system.models.schemas import GradeWorkflowOutput
from grade_system.services.llm_service import QwenWorkflowLLM
from grade_system.workflow.graph import build_grading_graph, extract_final_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the grading workflow on the 100-sample family-level evaluation set."
    )
    parser.add_argument(
        "--input",
        default=str(ROOT / "DataSet" / "family_evalset_100.jsonl"),
        help="Input JSONL dataset path.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "ExperimentResults" / "family_evalset_100_results.jsonl"),
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


def prepare_resume_state(
    output_path: Path,
) -> tuple[list[dict[str, Any]], set[str], dict[str, int]]:
    if not output_path.exists():
        return [], set(), {"success": 0, "error": 0}

    existing_results: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    status_counts = {"success": 0, "error": 0}
    truncate_pos: int | None = None

    with output_path.open("rb") as file:
        while True:
            line_start = file.tell()
            raw_line = file.readline()
            if not raw_line:
                break
            if not raw_line.strip():
                continue

            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, JSONDecodeError):
                truncate_pos = line_start
                break

            sample_id = str(record.get("sample_id", "")).strip()
            status = record.get("status")
            if status in status_counts:
                status_counts[status] += 1
            existing_results.append(record)
            if sample_id:
                completed_ids.add(sample_id)

    if truncate_pos is not None:
        with output_path.open("rb+") as file:
            file.truncate(truncate_pos)

    return existing_results, completed_ids, status_counts


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

    llm = QwenWorkflowLLM()
    llm.reset_usage_totals()

    try:
        graph = build_grading_graph(llm=llm)
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
    llm_usage = llm.get_usage_totals()

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
        "llm_usage": llm_usage,
        "workflow_output": workflow_output,
        "error": error_payload,
    }


def write_results(
    records: list[dict[str, Any]],
    output_path: Path,
    max_workers: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_results, completed_ids, status_counts = prepare_resume_state(output_path)
    pending_records = [
        record
        for record in records
        if str(record.get("id", "")).strip() not in completed_ids
    ]

    total = len(records)
    submitted = 0
    run_start_time = time.perf_counter()

    print(
        (
            f"Starting family eval run: total_samples={total}, "
            f"max_workers={max_workers}, output={output_path}"
        ),
        flush=True,
    )
    if existing_results or status_counts["error"]:
        print(
            (
                f"Resume detected: reusable_results={len(existing_results)} "
                f"(existing_success={status_counts['success']}, existing_error={status_counts['error']}) "
                f"remaining_samples={len(pending_records)}"
            ),
            flush=True,
        )

    if not pending_records:
        print(
            f"No pending samples. Existing results already cover all {total} dataset records.",
            flush=True,
        )
        return

    with output_path.open("a", encoding="utf-8", newline="\n") as output_file:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            record_iter = iter(pending_records)
            pending: dict[Future[dict[str, Any]], dict[str, Any]] = {}
            completed = len(existing_results)

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
                        f"Submitted sample {submitted}/{len(pending_records)}: "
                        f"id={next_record.get('id')} ref={next_record.get('reference_id')}"
                    ),
                    flush=True,
                )
                return True

            initial_window = min(max_workers, len(pending_records))
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
                            f"duration={result['timing']['duration_seconds']:.3f}s "
                            f"input_tokens={result['llm_usage']['input_tokens']} "
                            f"output_tokens={result['llm_usage']['output_tokens']}"
                        ),
                        flush=True,
                    )
                    submit_next()

    total_wall_time = time.perf_counter() - run_start_time
    print(
        (
            f"Family eval run finished. results_written_to={output_path} "
            f"total_wall_time_seconds={total_wall_time:.3f}"
        ),
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
