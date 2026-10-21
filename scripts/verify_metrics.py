"""Recalculate exported NTU metrics and verify historical numeric references."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from check_split_manifest import PACKAGE_ROOT, SPLITS, load_and_validate_inputs, read_json


METRIC_KEYS = ("ntu_mae", "ntu_rmse", "ntu_mape", "r2")
# Absolute tolerances cover measured float32 historical reduction differences.
HISTORICAL_ABS_TOLERANCE = {"ntu_mae": 1e-6, "ntu_rmse": 1e-5, "ntu_mape": 2e-6, "r2": 2e-8}
FLOAT64_ABS_TOLERANCE = 1e-9
FLOAT64_REL_TOLERANCE = 1e-12


def calculate_metrics(rows: list[dict[str, Any]], cap_mape: bool) -> dict[str, float]:
    """以双精度和 fsum 计算指标；分组 MAPE 不截断。"""
    if not rows:
        raise ValueError("Cannot calculate metrics for an empty group")
    targets = [row["target_ntu"] for row in rows]
    errors = [row["pred_ntu"] - row["target_ntu"] for row in rows]
    count = len(rows)
    squared_error = math.fsum(error * error for error in errors)
    target_mean = math.fsum(targets) / count
    total_variance = math.fsum((target - target_mean) ** 2 for target in targets)
    percentages = [abs(error) / (target + 1.0) * 100.0 for error, target in zip(errors, targets)]
    if cap_mape:
        percentages = [min(value, 1000.0) for value in percentages]
    values = {
        "ntu_mae": math.fsum(abs(error) for error in errors) / count,
        "ntu_rmse": math.sqrt(squared_error / count),
        "ntu_mape": math.fsum(percentages) / count,
        # Preserve the historical constant-target convention.
        "r2": 1.0 if total_variance == 0.0 else 1.0 - squared_error / total_variance,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Metric calculation produced a non-finite result")
    return values


def compare_metrics(actual: dict[str, Any], expected: dict[str, Any], label: str,
                    abs_tolerance: dict[str, float], rel_tolerance: float,
                    keys: tuple[str, ...] = METRIC_KEYS) -> list[dict[str, Any]]:
    """逐项核对外部历史值或已冻结的计算参考值。"""
    if set(expected) != set(keys):
        raise ValueError(f"Unexpected reference metric keys: {label}")
    checks: list[dict[str, Any]] = []
    for metric in keys:
        reference = expected[metric]
        if isinstance(reference, bool) or not isinstance(reference, (int, float)) or not math.isfinite(reference):
            raise ValueError(f"Invalid reference number: {label}, {metric}")
        observed = actual[metric]
        difference = abs(observed - reference)
        allowed = max(abs_tolerance[metric], rel_tolerance * max(abs(observed), abs(reference)))
        if difference > allowed:
            raise ValueError(f"Metric mismatch: {label}, {metric}: {observed:.17g} vs {reference:.17g}; "
                             f"absolute difference {difference:.6g} exceeds {allowed:.6g}")
        checks.append({"reference": label, "metric": metric, "actual": observed,
                       "expected": reference, "absolute_difference": difference,
                       "allowed_absolute_difference": allowed, "passed": True})
    return checks


def archived_zero_offset_mape(rows: list[dict[str, Any]]) -> float:
    """单独诊断旧分组表的零偏移分母，不替代公开指标。"""
    if any(row["target_ntu"] <= 0 for row in rows):
        raise ValueError("The archived zero-offset MAPE diagnostic requires positive targets")
    return math.fsum(abs(row["pred_ntu"] - row["target_ntu"]) / row["target_ntu"] * 100.0
                     for row in rows) / len(rows)


def verify(output_dir: Path) -> dict[str, Any]:
    """验证全部样本指标，并写入 JSON 与 CSV。"""
    rows, manifest, _ = load_and_validate_inputs()
    reference = read_json(PACKAGE_ROOT / "expected_results" / "metrics_reference.json")
    if reference.get("schema_version") != 1:
        raise ValueError("Unsupported metrics reference schema")
    if reference.get("historical_absolute_tolerance") != HISTORICAL_ABS_TOLERANCE:
        raise ValueError("Historical tolerances differ from the documented release tolerances")
    if reference.get("float64_absolute_tolerance") != FLOAT64_ABS_TOLERANCE or reference.get("float64_relative_tolerance") != FLOAT64_REL_TOLERANCE:
        raise ValueError("Float64 tolerances differ from the documented release tolerances")
    required_sections = {"historical_float32_split_metrics": {"val", "test"},
                         "archived_table_float64_split_metrics": set(SPLITS),
                         "exported_float64_split_metrics": set(SPLITS),
                         "exported_float64_group_metrics": {row["experiment_group"] for row in manifest},
                         "archived_zero_offset_group_metrics": {"10000NTU_100", "1000NTU_150", "20000NTU_200"}}
    for key, expected_names in required_sections.items():
        section = reference.get(key)
        if not isinstance(section, dict) or set(section) != expected_names:
            raise ValueError(f"Missing or unexpected reference entries: {key}")

    split_metrics: dict[str, dict[str, Any]] = {}
    group_metrics: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        subset = [row for row in rows if row["split"] == split]
        split_metrics[split] = {"samples": len(subset),
                               "groups": len({row["experiment_group"] for row in subset}),
                               **calculate_metrics(subset, cap_mape=True)}
    for group_row in manifest:
        group = group_row["experiment_group"]
        subset = [row for row in rows if row["experiment_group"] == group]
        group_metrics[group] = {"split": group_row["split"], "samples": len(subset),
                               **calculate_metrics(subset, cap_mape=False)}

    checks: list[dict[str, Any]] = []
    float64_tolerances = {key: FLOAT64_ABS_TOLERANCE for key in (*METRIC_KEYS, "ntu_mape_zero_offset")}
    for split in ("val", "test"):
        checks.extend(compare_metrics(split_metrics[split], reference["historical_float32_split_metrics"][split],
                                      f"historical_float32/{split}", HISTORICAL_ABS_TOLERANCE, 0.0))
    for split in SPLITS:
        checks.extend(compare_metrics(split_metrics[split], reference["archived_table_float64_split_metrics"][split],
                                      f"archived_table_float64/{split}", float64_tolerances, FLOAT64_REL_TOLERANCE,
                                      keys=("ntu_mae", "ntu_rmse")))
        checks.extend(compare_metrics(split_metrics[split], reference["exported_float64_split_metrics"][split],
                                      f"exported_float64/{split}", float64_tolerances, FLOAT64_REL_TOLERANCE))
    for group in group_metrics:
        checks.extend(compare_metrics(group_metrics[group], reference["exported_float64_group_metrics"][group],
                                      f"exported_float64/{group}", float64_tolerances, FLOAT64_REL_TOLERANCE))
    archived_diagnostics: dict[str, dict[str, float]] = {}
    for group, metrics in reference["archived_zero_offset_group_metrics"].items():
        subset = [row for row in rows if row["experiment_group"] == group]
        archived_diagnostics[group] = {"ntu_mae": group_metrics[group]["ntu_mae"],
                                       "ntu_mape_zero_offset": archived_zero_offset_mape(subset),
                                       "released_ntu_mape_target_plus_one": group_metrics[group]["ntu_mape"]}
        checks.extend(compare_metrics(archived_diagnostics[group], metrics, f"archived_group_zero_offset/{group}",
                                      float64_tolerances, FLOAT64_REL_TOLERANCE,
                                      keys=("ntu_mae", "ntu_mape_zero_offset")))

    result = {"status": "passed", "verified_comparisons": len(checks),
              "aggregation": "Python float (binary64) with math.fsum from exported NTU columns",
              "split_mape": "mean(min(abs(pred-target)/(target+1)*100, 1000))",
              "group_mape": "mean(abs(pred-target)/(target+1)*100), uncapped",
              "split_metrics": split_metrics, "group_metrics": group_metrics,
              "archived_group_diagnostic_formula": "mean(abs(pred-target)/target*100), uncapped; a separate zero-offset diagnostic matching archived group summaries, not the released target+1 group MAPE",
              "archived_group_zero_offset_diagnostics": archived_diagnostics,
              "comparisons": checks}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_verification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for filename, identity, collection, extra in (
        ("split_metrics.csv", "split", split_metrics, ["samples", "groups"]),
        ("group_metrics.csv", "experiment_group", group_metrics, ["split", "samples"]),
    ):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[identity, *extra, *METRIC_KEYS], lineterminator="\n")
            writer.writeheader()
            writer.writerows({identity: key, **value} for key, value in collection.items())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "verification_output")
    args = parser.parse_args()
    result = verify(args.output_dir)
    print(f"Metric verification passed: {result['verified_comparisons']} numeric comparisons.")


if __name__ == "__main__":
    main()
