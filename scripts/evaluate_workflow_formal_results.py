from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ERROR_LABELS = [
    "公式与法则误用",
    "基础计算失误",
    "审题遗漏与条件忽视",
    "概念混淆与理解偏差",
    "逻辑推断失误",
]
MISSING_PRIMARY_LABEL = "未输出主标签"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate workflow experiment results on whole-problem correctness, "
            "error attribution, and runtime success rate."
        )
    )
    parser.add_argument(
        "--input",
        default=str(ROOT / "ExperimentResults" / "family_evalset_100_results.jsonl"),
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


def collect_runtime_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_records = len(records)
    success_count = sum(1 for record in records if record.get("status") == "success")
    error_count = total_records - success_count

    return {
        "total_records": total_records,
        "success_count": success_count,
        "error_count": error_count,
        "success_rate": safe_divide(success_count, total_records),
    }


def normalize_binary_label(value: Any) -> bool:
    return bool(value)


def compute_grading_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible_records = [
        record
        for record in records
        if record.get("gold", {}).get("is_correct") is not None
    ]

    failed_record_count = 0
    normalized_pairs: list[tuple[bool, bool]] = []
    for record in eligible_records:
        gold_label = normalize_binary_label(record["gold"]["is_correct"])
        if (
            record.get("status") == "success"
            and record.get("prediction", {}).get("is_correct") is not None
        ):
            pred_label = normalize_binary_label(record["prediction"]["is_correct"])
        else:
            failed_record_count += 1
            pred_label = not gold_label
        normalized_pairs.append((gold_label, pred_label))

    tp = sum(
        1
        for gold_label, pred_label in normalized_pairs
        if gold_label is True and pred_label is True
    )
    fp = sum(
        1
        for gold_label, pred_label in normalized_pairs
        if gold_label is False and pred_label is True
    )
    fn = sum(
        1
        for gold_label, pred_label in normalized_pairs
        if gold_label is True and pred_label is False
    )
    tn = sum(
        1
        for gold_label, pred_label in normalized_pairs
        if gold_label is False and pred_label is False
    )

    sample_count = len(eligible_records)
    accuracy = safe_divide(tp + tn, sample_count)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = f1_score(precision, recall)

    return {
        "evaluation_scope": "统计全部样本；对运行失败样本，按整题判定错误计入分类指标。",
        "positive_class": "整题正确",
        "sample_count": sample_count,
        "failed_records_counted_as_wrong": failed_record_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def normalize_primary_reason_type(value: Any) -> str:
    normalized = (value or "").strip()
    return normalized or MISSING_PRIMARY_LABEL


def build_confusion_matrix(
    gold_labels: list[str],
    pred_labels: list[str],
    row_order: list[str],
) -> dict[str, dict[str, int]]:
    extra_predicted_labels = [
        label
        for label in dict.fromkeys(pred_labels)
        if label not in row_order
    ]
    columns = [*row_order, *extra_predicted_labels]

    matrix: dict[str, dict[str, int]] = {}
    for gold_label in row_order:
        row: dict[str, int] = {}
        for pred_label in columns:
            row[pred_label] = sum(
                1
                for gold, pred in zip(gold_labels, pred_labels)
                if gold == gold_label and pred == pred_label
            )
        matrix[gold_label] = row
    return matrix


def compute_attribution_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible_negative_records = [
        record
        for record in records
        if record.get("gold", {}).get("is_correct") is False
    ]

    gold_labels = [
        str(record.get("gold", {}).get("error_category", "")).strip()
        for record in eligible_negative_records
    ]
    pred_labels: list[str] = []
    failed_negative_count = 0
    for record in eligible_negative_records:
        if record.get("status") == "success":
            pred_labels.append(
                normalize_primary_reason_type(
                    record.get("prediction", {}).get("primary_reason_type")
                )
            )
        else:
            failed_negative_count += 1
            pred_labels.append(MISSING_PRIMARY_LABEL)

    per_label_metrics: dict[str, dict[str, float | int]] = {}
    for label in ERROR_LABELS:
        tp = sum(
            1
            for gold, pred in zip(gold_labels, pred_labels)
            if gold == label and pred == label
        )
        fp = sum(
            1
            for gold, pred in zip(gold_labels, pred_labels)
            if gold != label and pred == label
        )
        fn = sum(
            1
            for gold, pred in zip(gold_labels, pred_labels)
            if gold == label and pred != label
        )
        support = sum(1 for gold in gold_labels if gold == label)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        per_label_metrics[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1_score(precision, recall),
        }

    sample_count = len(eligible_negative_records)
    accuracy = safe_divide(
        sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == pred),
        sample_count,
    )
    macro_precision = safe_divide(
        sum(metric["precision"] for metric in per_label_metrics.values()),
        len(ERROR_LABELS),
    )
    macro_recall = safe_divide(
        sum(metric["recall"] for metric in per_label_metrics.values()),
        len(ERROR_LABELS),
    )
    macro_f1 = safe_divide(
        sum(metric["f1"] for metric in per_label_metrics.values()),
        len(ERROR_LABELS),
    )
    primary_reason_type_coverage = safe_divide(
        sum(
            1
            for record in eligible_negative_records
            if str(
                record.get("prediction", {}).get("primary_reason_type", "")
            ).strip()
        ),
        sample_count,
    )

    return {
        "evaluation_scope": "统计全部反例样本；对运行失败反例，按错误归因错误计入分类指标。",
        "sample_count": sample_count,
        "eligible_negative_records": len(eligible_negative_records),
        "successful_negative_records": len(eligible_negative_records) - failed_negative_count,
        "failed_negative_records_counted_as_wrong": failed_negative_count,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "primary_reason_type_coverage": primary_reason_type_coverage,
        "per_label_metrics": per_label_metrics,
        "confusion_matrix": build_confusion_matrix(
            gold_labels=gold_labels,
            pred_labels=pred_labels,
            row_order=ERROR_LABELS,
        ),
    }


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "metric_definition": {
            "grading_task": "整题层面判断学生作答是否正确",
            "attribution_task": "当样本为反例时，识别其主要错误类型",
            "runtime_task": "统计工作流成功返回结果的样本比例",
        },
        "runtime_effectiveness": collect_runtime_metrics(records),
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
