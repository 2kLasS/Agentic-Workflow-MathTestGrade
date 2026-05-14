from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate workflow experiment results for grading and attribution metrics."
    )
    parser.add_argument(
        "--input",
        default=str(ROOT / "ExperimentResults" / "pilot_testset_30_results.jsonl"),
        help="Input JSONL results path.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON summary output path.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the summary JSON.",
    )
    return parser.parse_args()


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_label_metrics(
    gold_labels: list[str],
    pred_labels: list[str],
    label_order: list[str],
) -> dict[str, Any]:
    total = len(gold_labels)
    correct = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == pred)

    confusion_matrix: dict[str, dict[str, int]] = {}
    predicted_label_order = [label for label in dict.fromkeys(pred_labels) if label not in label_order]
    matrix_columns = [*label_order, *predicted_label_order]

    for gold_label in label_order:
        row: dict[str, int] = {}
        for pred_label in matrix_columns:
            row[pred_label] = sum(
                1
                for gold, pred in zip(gold_labels, pred_labels)
                if gold == gold_label and pred == pred_label
            )
        confusion_matrix[gold_label] = row

    label_metrics: dict[str, dict[str, float | int]] = {}
    for label in label_order:
        tp = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == label and pred != label)
        support = sum(1 for gold in gold_labels if gold == label)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        label_metrics[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1_score(precision, recall),
        }

    macro_precision = safe_divide(
        sum(metric["precision"] for metric in label_metrics.values()),
        len(label_metrics),
    )
    macro_recall = safe_divide(
        sum(metric["recall"] for metric in label_metrics.values()),
        len(label_metrics),
    )
    macro_f1 = safe_divide(
        sum(metric["f1"] for metric in label_metrics.values()),
        len(label_metrics),
    )

    return {
        "sample_count": total,
        "accuracy": safe_divide(correct, total),
        "label_metrics": label_metrics,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion_matrix,
    }


def compute_grading_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful_records = [record for record in records if record.get("status") == "success"]
    label_name = {True: "correct", False: "incorrect"}

    gold_labels = [label_name[bool(record["gold"]["is_correct"])] for record in successful_records]
    pred_labels = [
        label_name[bool(record["prediction"]["is_correct"])]
        for record in successful_records
    ]
    metrics = compute_label_metrics(
        gold_labels=gold_labels,
        pred_labels=pred_labels,
        label_order=["correct", "incorrect"],
    )

    exact_matches_all_records = sum(
        1
        for record in records
        if record.get("status") == "success"
        and record["gold"]["is_correct"] == record["prediction"]["is_correct"]
    )

    duration_values = [
        record["timing"]["duration_seconds"]
        for record in successful_records
        if record.get("timing", {}).get("duration_seconds") is not None
    ]

    return {
        "total_records": len(records),
        "successful_records": len(successful_records),
        "failed_records": len(records) - len(successful_records),
        "success_rate": safe_divide(len(successful_records), len(records)),
        "all_record_exact_match_rate": safe_divide(exact_matches_all_records, len(records)),
        "successful_record_metrics": metrics,
        "timing_seconds": {
            "mean": statistics.mean(duration_values) if duration_values else 0.0,
            "median": statistics.median(duration_values) if duration_values else 0.0,
            "max": max(duration_values) if duration_values else 0.0,
            "min": min(duration_values) if duration_values else 0.0,
        },
    }


def compute_multilabel_list_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    contains_gold_count = 0
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    predicted_label_counts: list[int] = []

    for record in records:
        gold_label = record["gold"]["error_category"]
        pred_list = [
            item.strip()
            for item in record["prediction"].get("reason_type", [])
            if item and item.strip()
        ]
        pred_set = set(pred_list)
        contains_gold = gold_label in pred_set
        if contains_gold:
            contains_gold_count += 1

        precision = safe_divide(1.0 if contains_gold else 0.0, len(pred_set))
        recall = 1.0 if contains_gold else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1_score(precision, recall))
        predicted_label_counts.append(len(pred_set))

    total = len(records)
    return {
        "sample_count": total,
        "contains_gold_accuracy": safe_divide(contains_gold_count, total),
        "sample_avg_precision": statistics.mean(precision_values) if precision_values else 0.0,
        "sample_avg_recall": statistics.mean(recall_values) if recall_values else 0.0,
        "sample_avg_f1": statistics.mean(f1_values) if f1_values else 0.0,
        "average_predicted_label_count": (
            statistics.mean(predicted_label_counts) if predicted_label_counts else 0.0
        ),
    }


def compute_attribution_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    gold_error_records = [record for record in records if record["gold"]["is_correct"] is False]
    successful_error_records = [
        record for record in gold_error_records if record.get("status") == "success"
    ]

    gold_labels = [record["gold"]["error_category"] for record in successful_error_records]
    pred_labels = [
        (record["prediction"].get("primary_reason_type") or "").strip() or "未输出主标签"
        for record in successful_error_records
    ]
    label_order = sorted({record["gold"]["error_category"] for record in successful_error_records})

    primary_metrics = compute_label_metrics(
        gold_labels=gold_labels,
        pred_labels=pred_labels,
        label_order=label_order,
    )
    list_metrics = compute_multilabel_list_metrics(successful_error_records)
    primary_prediction_coverage = safe_divide(
        sum(
            1
            for record in successful_error_records
            if (record["prediction"].get("primary_reason_type") or "").strip()
        ),
        len(successful_error_records),
    )

    return {
        "eligible_error_records": len(gold_error_records),
        "successful_error_records": len(successful_error_records),
        "failed_error_records": len(gold_error_records) - len(successful_error_records),
        "primary_reason_type_coverage": primary_prediction_coverage,
        "primary_reason_type_metrics": primary_metrics,
        "reason_type_list_metrics": list_metrics,
    }


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_summary": {
            "total_records": len(records),
            "success_count": sum(1 for record in records if record.get("status") == "success"),
            "error_count": sum(1 for record in records if record.get("status") != "success"),
        },
        "grading_metrics": compute_grading_metrics(records),
        "attribution_metrics": compute_attribution_metrics(records),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")

    records = load_jsonl_records(input_path)
    if not records:
        raise ValueError(f"No records found in results file: {input_path}")

    summary = build_summary(records)
    summary_json = json.dumps(
        summary,
        ensure_ascii=False,
        indent=2 if args.pretty else None,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(summary_json, encoding="utf-8")

    print(summary_json)


if __name__ == "__main__":
    main()
