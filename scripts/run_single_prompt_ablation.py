from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from single_prompt_ablation_workflow import (  # noqa: E402
    QWEN_MODEL,
    QWEN_TEMPERATURE,
    build_prediction,
    call_single_prompt_model,
    parse_single_prompt_output,
)

DEFAULT_INPUT_PATH = SCRIPT_DIR / "family_evalset_100.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the single-prompt grading ablation on a JSONL dataset using "
            "sliding-window multithreading."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Input JSONL dataset path. Defaults to family_evalset_100.jsonl next to this script.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSONL path. Defaults to <input_stem>_single_prompt_ablation_results.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional summary JSON path. Defaults to <input_stem>_single_prompt_ablation_summary.json",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent model calls.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on dataset rows to run. 0 means no limit.",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_single_prompt_ablation_results.jsonl")


def default_summary_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_single_prompt_ablation_summary.json")


def load_jsonl_records(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_index, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record["_dataset_index"] = line_index
            records.append(record)
            if limit and len(records) >= limit:
                break
    return records


def load_existing_results(path: Path) -> tuple[list[dict[str, Any]], set[int]]:
    if not path.exists():
        return [], set()

    existing_results: list[dict[str, Any]] = []
    completed_dataset_indexes: set[int] = set()

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            result = json.loads(line)
            existing_results.append(result)
            dataset_index = result.get("dataset_index")
            if isinstance(dataset_index, int):
                completed_dataset_indexes.add(dataset_index)

    return existing_results, completed_dataset_indexes


def merge_results_by_dataset_index(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered_results: dict[int, dict[str, Any]] = {}
    non_indexed_results: list[dict[str, Any]] = []

    for result in results:
        dataset_index = result.get("dataset_index")
        if isinstance(dataset_index, int):
            if dataset_index not in ordered_results:
                ordered_results[dataset_index] = result
            continue
        non_indexed_results.append(result)

    merged_results = [ordered_results[key] for key in sorted(ordered_results)]
    merged_results.extend(non_indexed_results)
    return merged_results


def _safe_total_tokens(tokens: dict[str, Any]) -> int | None:
    total_tokens = tokens.get("total_tokens")
    if total_tokens is not None:
        return total_tokens

    input_tokens = tokens.get("input_tokens")
    output_tokens = tokens.get("output_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    return input_tokens + output_tokens


def _is_truncated_finish_reason(finish_reason: Any) -> bool:
    normalized = str(finish_reason or "").strip().lower()
    return normalized in {"length", "max_tokens"}


def run_single_record(record: dict[str, Any]) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    total_start_time = time.perf_counter()

    workflow_input = {
        "question_text": record["question"],
        "student_answer_text": record["answer"],
    }

    llm_payload: dict[str, Any] | None = None
    status = "success"
    analysis_output: dict[str, Any] | None = None
    prediction: dict[str, Any] | None = None
    error_payload: dict[str, Any] | None = None
    parse_duration_seconds = 0.0

    try:
        llm_payload = call_single_prompt_model(
            question_text=workflow_input["question_text"],
            student_answer_text=workflow_input["student_answer_text"],
        )
    except Exception as exc:  # pragma: no cover - runtime protection
        status = "api_error"
        error_payload = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    if llm_payload is not None:
        raw_text = llm_payload["raw_text"]
        finish_reason = llm_payload.get("finish_reason")
        parse_start_time = time.perf_counter()
        try:
            parsed_output = parse_single_prompt_output(raw_text)
            parse_duration_seconds = round(time.perf_counter() - parse_start_time, 6)
            analysis_output = parsed_output.model_dump()
            prediction = build_prediction(parsed_output)
            status = "success"
        except ValidationError as exc:
            parse_duration_seconds = round(time.perf_counter() - parse_start_time, 6)
            status = "truncated_output" if _is_truncated_finish_reason(finish_reason) else "validation_error"
            error_payload = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        except Exception as exc:  # pragma: no cover - runtime protection
            parse_duration_seconds = round(time.perf_counter() - parse_start_time, 6)
            status = "truncated_output" if _is_truncated_finish_reason(finish_reason) else "parse_error"
            error_payload = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }

    if prediction is None:
        prediction = {
            "is_correct": None,
            "incorrect_subproblem_indices": [],
            "autonomous_error_count": 0,
            "overall_feedback_text": "",
            "error_reports": [],
        }

    finished_at = datetime.now(timezone.utc)
    total_duration_seconds = round(time.perf_counter() - total_start_time, 6)

    llm_timing = (llm_payload or {}).get("timing", {})
    tokens = (llm_payload or {}).get(
        "tokens",
        {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "raw_token_usage": {},
        },
    )
    tokens["total_tokens"] = _safe_total_tokens(tokens)

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
        "analysis_output": analysis_output,
        "raw_model_output_text": (llm_payload or {}).get("raw_text", ""),
        "llm": {
            "finish_reason": (llm_payload or {}).get("finish_reason"),
            "is_truncated": _is_truncated_finish_reason((llm_payload or {}).get("finish_reason")),
            "response_metadata": (llm_payload or {}).get("response_metadata", {}),
            "usage_metadata": (llm_payload or {}).get("usage_metadata", {}),
        },
        "tokens": tokens,
        "timing": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "model_call_duration_seconds": llm_timing.get("model_call_duration_seconds"),
            "parse_duration_seconds": parse_duration_seconds,
            "total_duration_seconds": total_duration_seconds,
        },
        "error": error_payload,
    }


def build_summary(
    results: list[dict[str, Any]],
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    run_started_at: datetime,
    run_finished_at: datetime,
    total_wall_time_seconds: float,
    max_workers: int,
    existing_result_count: int,
    resumed_skip_count: int,
    newly_run_count: int,
) -> dict[str, Any]:
    status_counts = Counter(result["status"] for result in results)
    finish_reason_counts = Counter(
        (result.get("llm", {}).get("finish_reason") or "unknown")
        for result in results
    )

    total_duration_values = [
        result["timing"]["total_duration_seconds"]
        for result in results
        if result.get("timing", {}).get("total_duration_seconds") is not None
    ]
    model_call_duration_values = [
        result["timing"]["model_call_duration_seconds"]
        for result in results
        if result.get("timing", {}).get("model_call_duration_seconds") is not None
    ]
    parse_duration_values = [
        result["timing"]["parse_duration_seconds"]
        for result in results
        if result.get("timing", {}).get("parse_duration_seconds") is not None
    ]

    input_token_values = [
        result["tokens"]["input_tokens"]
        for result in results
        if result.get("tokens", {}).get("input_tokens") is not None
    ]
    output_token_values = [
        result["tokens"]["output_tokens"]
        for result in results
        if result.get("tokens", {}).get("output_tokens") is not None
    ]
    total_token_values = [
        result["tokens"]["total_tokens"]
        for result in results
        if result.get("tokens", {}).get("total_tokens") is not None
    ]

    success_count = status_counts.get("success", 0)
    total_count = len(results)

    return {
        "run_summary": {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "summary_path": str(summary_path),
            "model": QWEN_MODEL,
            "temperature": QWEN_TEMPERATURE,
            "max_workers": max_workers,
            "total_records": total_count,
            "success_count": success_count,
            "failure_count": total_count - success_count,
            "success_rate": (success_count / total_count) if total_count else 0.0,
            "existing_result_count_before_run": existing_result_count,
            "resumed_skip_count": resumed_skip_count,
            "newly_run_count": newly_run_count,
            "status_counts": dict(status_counts),
            "finish_reason_counts": dict(finish_reason_counts),
            "started_at": run_started_at.isoformat(),
            "finished_at": run_finished_at.isoformat(),
            "total_wall_time_seconds": round(total_wall_time_seconds, 6),
        },
        "timing_seconds": {
            "total_duration_mean": statistics.mean(total_duration_values) if total_duration_values else 0.0,
            "total_duration_median": statistics.median(total_duration_values) if total_duration_values else 0.0,
            "model_call_mean": statistics.mean(model_call_duration_values) if model_call_duration_values else 0.0,
            "model_call_median": statistics.median(model_call_duration_values) if model_call_duration_values else 0.0,
            "parse_mean": statistics.mean(parse_duration_values) if parse_duration_values else 0.0,
            "parse_median": statistics.median(parse_duration_values) if parse_duration_values else 0.0,
        },
        "tokens": {
            "records_with_input_tokens": len(input_token_values),
            "records_with_output_tokens": len(output_token_values),
            "records_with_total_tokens": len(total_token_values),
            "sum_input_tokens": sum(input_token_values),
            "sum_output_tokens": sum(output_token_values),
            "sum_total_tokens": sum(total_token_values),
            "avg_input_tokens": statistics.mean(input_token_values) if input_token_values else 0.0,
            "avg_output_tokens": statistics.mean(output_token_values) if output_token_values else 0.0,
            "avg_total_tokens": statistics.mean(total_token_values) if total_token_values else 0.0,
        },
    }


def write_results(
    records: list[dict[str, Any]],
    output_path: Path,
    max_workers: int,
) -> tuple[list[dict[str, Any]], datetime, datetime, float, int, int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_results, completed_dataset_indexes = load_existing_results(output_path)
    pending_records = [
        record
        for record in records
        if record["_dataset_index"] not in completed_dataset_indexes
    ]
    skipped_count = len(records) - len(pending_records)
    total = len(pending_records)
    submitted = 0
    run_started_at = datetime.now(timezone.utc)
    run_start_time = time.perf_counter()
    written_results: list[dict[str, Any]] = list(existing_results)

    print(
        (
            f"Starting single-prompt ablation run: total_samples={len(records)}, "
            f"pending_samples={total}, resumed_skips={skipped_count}, "
            f"max_workers={max_workers}, output={output_path}, "
            f"model={QWEN_MODEL}, temperature={QWEN_TEMPERATURE}"
        ),
        flush=True,
    )

    if total == 0:
        total_wall_time_seconds = time.perf_counter() - run_start_time
        run_finished_at = datetime.now(timezone.utc)
        print(
            "No pending samples found. Existing results already cover the requested dataset.",
            flush=True,
        )
        return (
            merge_results_by_dataset_index(written_results),
            run_started_at,
            run_finished_at,
            total_wall_time_seconds,
            len(existing_results),
            skipped_count,
            0,
        )

    with output_path.open("a", encoding="utf-8", newline="\n") as output_file:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            record_iter = iter(pending_records)
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
                    written_results.append(result)
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()
                    completed += 1
                    print(
                        (
                            f"[{completed}/{total}] "
                            f"id={source_record.get('id')} "
                            f"status={result['status']} "
                            f"finish_reason={result['llm']['finish_reason']} "
                            f"total_tokens={result['tokens']['total_tokens']} "
                            f"duration={result['timing']['total_duration_seconds']:.3f}s"
                        ),
                        flush=True,
                    )
                    submit_next()

    total_wall_time_seconds = time.perf_counter() - run_start_time
    run_finished_at = datetime.now(timezone.utc)
    print(
        (
            "Single-prompt ablation run finished. "
            f"results_written_to={output_path} "
            f"total_wall_time_seconds={total_wall_time_seconds:.3f}"
        ),
        flush=True,
    )

    return (
        merge_results_by_dataset_index(written_results),
        run_started_at,
        run_finished_at,
        total_wall_time_seconds,
        len(existing_results),
        skipped_count,
        total,
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)
    summary_path = Path(args.summary_output) if args.summary_output else default_summary_path(input_path)

    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1.")
    if args.limit < 0:
        raise ValueError("--limit must be greater than or equal to 0.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    records = load_jsonl_records(input_path, limit=args.limit)
    if not records:
        raise ValueError(f"No records found in dataset: {input_path}")

    (
        results,
        run_started_at,
        run_finished_at,
        total_wall_time_seconds,
        existing_result_count,
        resumed_skip_count,
        newly_run_count,
    ) = write_results(
        records=records,
        output_path=output_path,
        max_workers=args.max_workers,
    )
    summary = build_summary(
        results=results,
        input_path=input_path,
        output_path=output_path,
        summary_path=summary_path,
        run_started_at=run_started_at,
        run_finished_at=run_finished_at,
        total_wall_time_seconds=total_wall_time_seconds,
        max_workers=args.max_workers,
        existing_result_count=existing_result_count,
        resumed_skip_count=resumed_skip_count,
        newly_run_count=newly_run_count,
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
