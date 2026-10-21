"""Check the released experiment-group split without training dependencies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
PREDICTION_FIELDS = ["experiment_group", "frame_index", "split", "pred_ntu", "target_ntu"]
MANIFEST_FIELDS = ["experiment_group", "split", "samples"]
SAMPLES_PER_GROUP = 599
EXPECTED_GROUP_COUNTS = {"train": 12, "val": 1, "test": 2}
EXPECTED_SAMPLE_COUNTS = {"train": 7188, "val": 599, "test": 1198}
EXPECTED_ASSIGNMENTS = {
    f"{ntu}NTU_{dose}": (
        "val" if (ntu, dose) == (10000, 100)
        else "test" if (ntu, dose) in {(1000, 150), (20000, 200)}
        else "train"
    )
    for ntu in (1000, 10000, 20000)
    for dose in (10, 50, 100, 150, 200)
}


def sha256(path: Path) -> str:
    """计算文件摘要。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """拒绝重复键和非有限 JSON 数值。"""
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value in {path.name}: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=reject_constant, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return value


def read_csv(path: Path, fields: list[str]) -> list[dict[str, str]]:
    """逐行检查 CSV，不丢弃空行或缺失字段。"""
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        if next(reader, None) != fields:
            raise ValueError(f"Unexpected columns in {path.name}; expected {fields}")
        for line_number, values in enumerate(reader, start=2):
            if len(values) != len(fields) or any(value == "" for value in values):
                raise ValueError(f"Missing or extra CSV values in {path.name}, record {line_number}")
            if any(value != value.strip() for value in values):
                raise ValueError(f"Unexpected surrounding whitespace in {path.name}, record {line_number}")
            records.append(dict(zip(fields, values)))
    if not records:
        raise ValueError(f"No data rows in {path.name}")
    return records


def integer(value: str, label: str) -> int:
    """仅接受规范整数文本。"""
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {label}: {value!r}") from exc
    if str(result) != value:
        raise ValueError(f"Non-canonical integer for {label}: {value!r}")
    return result


def load_and_validate_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """核对固定分组、样本标识及所有有限数值。"""
    reference = read_json(PACKAGE_ROOT / "expected_results" / "split_reference.json")
    expected_protocol = {
        "schema_version": 1,
        "total_samples": 8985,
        "total_groups": 15,
        "samples_per_group": SAMPLES_PER_GROUP,
        "frame_index_min": 1,
        "frame_index_max": 599,
        "split_group_counts": EXPECTED_GROUP_COUNTS,
        "split_sample_counts": EXPECTED_SAMPLE_COUNTS,
        "group_assignments": EXPECTED_ASSIGNMENTS,
        "group_sample_counts": {group: SAMPLES_PER_GROUP for group in EXPECTED_ASSIGNMENTS},
    }
    for key, expected in expected_protocol.items():
        if reference.get(key) != expected:
            raise ValueError(f"Split reference does not match the formal protocol: {key}")

    manifest: list[dict[str, Any]] = []
    assignments: dict[str, str] = {}
    for row in read_csv(PACKAGE_ROOT / "data" / "group_manifest.csv", MANIFEST_FIELDS):
        group = row["experiment_group"]
        if group in assignments:
            raise ValueError(f"Duplicate group in manifest: {group}")
        samples = integer(row["samples"], f"{group} samples")
        if row["split"] not in SPLITS or samples != SAMPLES_PER_GROUP:
            raise ValueError(f"Invalid manifest split or sample count: {group}")
        assignments[group] = row["split"]
        manifest.append({"experiment_group": group, "split": row["split"], "samples": samples})
    if assignments != EXPECTED_ASSIGNMENTS:
        raise ValueError("The group manifest does not match all 15 formal group assignments")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    frames: dict[str, set[int]] = defaultdict(set)
    split_groups: dict[str, set[str]] = {split: set() for split in SPLITS}
    split_samples: Counter[str] = Counter()
    for row in read_csv(PACKAGE_ROOT / "data" / "mainline_predictions.csv", PREDICTION_FIELDS):
        group, split = row["experiment_group"], row["split"]
        if group not in assignments or split != assignments[group]:
            raise ValueError(f"Prediction split disagrees with manifest: {group}, {split}")
        frame_index = integer(row["frame_index"], f"{group} frame_index")
        identity = (group, frame_index)
        if identity in seen:
            raise ValueError(f"Duplicate sample identity: {group}, frame {frame_index}")
        seen.add(identity)
        numeric: dict[str, float] = {}
        for field in ("pred_ntu", "target_ntu"):
            try:
                value = float(row[field])
            except ValueError as exc:
                raise ValueError(f"Invalid {field}: {identity}") from exc
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {field}: {identity}")
            numeric[field] = value
        if numeric["target_ntu"] < 0:
            raise ValueError(f"Negative target_ntu: {identity}")
        frames[group].add(frame_index)
        split_groups[split].add(group)
        split_samples[split] += 1
        rows.append({"experiment_group": group, "frame_index": frame_index,
                     "split": split, **numeric})

    expected_frames = set(range(1, SAMPLES_PER_GROUP + 1))
    if set(frames) != set(EXPECTED_ASSIGNMENTS):
        raise ValueError("Prediction groups do not match the manifest")
    for group in EXPECTED_ASSIGNMENTS:
        if frames[group] != expected_frames:
            raise ValueError(f"Expected frame indices 1 through 599 exactly once: {group}")
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            if split_groups[left] & split_groups[right]:
                raise ValueError(f"Experiment groups overlap between {left} and {right}")
    group_counts = {split: len(split_groups[split]) for split in SPLITS}
    if group_counts != EXPECTED_GROUP_COUNTS or dict(split_samples) != EXPECTED_SAMPLE_COUNTS:
        raise ValueError("Split group or sample counts do not match the formal protocol")
    if len(rows) != 8985 or len(seen) != 8985:
        raise ValueError("Expected 8,985 distinct sample identities")
    return rows, manifest, reference


def verify(output_dir: Path) -> dict[str, Any]:
    """验证后仅在指定输出目录写入结果。"""
    rows, manifest, _ = load_and_validate_inputs()
    summary = [{"split": split, "groups": sum(row["split"] == split for row in manifest),
                "samples": sum(row["split"] == split for row in rows)} for split in SPLITS]
    result = {
        "status": "passed", "total_samples": len(rows), "total_groups": len(manifest),
        "unique_sample_identities": len(rows), "samples_per_group": 599,
        "frame_index_min": 1, "frame_index_max": 599, "group_intersections": [],
        "split_summary": summary, "group_manifest": manifest,
        "input_sha256": {
            "mainline_predictions.csv": sha256(PACKAGE_ROOT / "data" / "mainline_predictions.csv"),
            "group_manifest.csv": sha256(PACKAGE_ROOT / "data" / "group_manifest.csv"),
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_verification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (output_dir / "split_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "groups", "samples"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "verification_output")
    args = parser.parse_args()
    result = verify(args.output_dir)
    print(f"Split verification passed: {result['total_samples']} samples across {result['total_groups']} groups.")


if __name__ == "__main__":
    main()
