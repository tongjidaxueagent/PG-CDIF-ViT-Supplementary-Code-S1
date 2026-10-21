"""Verify nine disclosed side-view attention distances with Python >= 3.9.

Only the standard library and the two bundled data/reference files are used.
This verifies saved coordinates, their normalization, and arithmetic means. It
does not generate attention maps, run model inference, or establish the origin
or correctness of the interface labels. Six frames belong to the test split;
three frames from 10000NTU_100 are validation references (split ``val``).

The heatmap peak is zero-based on a 224-row grid, so its denominator is 223.
The interface uses the original side-view ROI height as its denominator.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PACKAGE_DIR / "data" / "attention_distance_samples.csv"
REFERENCE_PATH = PACKAGE_DIR / "expected_results" / "attention_distance_reference.json"
ABS_TOL = 1e-12
GROUPS = (("1000NTU_150", "test", 743), ("10000NTU_100", "val", 760),
          ("20000NTU_200", "test", 788))
PHASES = (("early", 61), ("middle", 300), ("late", 539))
FIELDS = ("sample_id", "group", "split", "phase", "frame_index", "view",
          "peak_row", "image_height", "interface_position_pixels", "roi_height",
          "saved_peak_y_norm", "saved_interface_y_norm", "saved_distance")


def require(condition, message):
    """Raise a reproducible, actionable verification failure."""
    if not condition:
        raise ValueError(message)


def number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid numeric value for " + name) from error
    require(math.isfinite(result), "Non-finite value for " + name)
    return result


def integer(value, name):
    result = number(value, name)
    require(result.is_integer(), "Non-integer value for " + name)
    return int(result)


def close(actual, expected, name):
    expected_value = number(expected, name + " reference")
    require(math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=ABS_TOL),
            f"{name}: computed {actual!r}, expected {expected_value!r}")


def verify(output_dir: Path) -> dict:
    """Check the bundled samples against frozen source values and write results.

    ``output_dir`` may be absolute or relative to the caller's working directory.
    All input paths are relative to this script, so moving the package is safe.
    A mismatch raises ValueError before any result files are written.
    """
    with DATA_PATH.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == list(FIELDS), "Unexpected attention-distance CSV schema")
        rows = list(reader)
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8-sig"))
    require(reference.get("schema_version") == 1, "Unexpected reference schema")
    require(reference.get("absolute_tolerance") == ABS_TOL, "Unexpected reference tolerance")
    require(reference.get("sample_count") == len(rows) == 9, "Expected exactly nine side frames")
    require(reference.get("image_height") == 224, "Expected 224-row attention grid")
    require(reference.get("split_counts") == {"test": 6, "val": 3}, "Unexpected reference split counts")
    expected_identities = {
        f"{group}_side_{frame}": (group, split, phase, frame, roi_height)
        for group, split, roi_height in GROUPS for phase, frame in PHASES
    }
    observed_ids = [row["sample_id"] for row in rows]
    require(len(set(observed_ids)) == len(rows), "Duplicate sample IDs")
    require(set(observed_ids) == set(expected_identities), "Missing or unexpected sample IDs")
    split_counts = {split: sum(row["split"] == split for row in rows) for split in ("test", "val")}
    require(split_counts == reference["split_counts"], "Expected six test and three validation frames")
    recomputed = []
    for row in rows:
        sample_id = row["sample_id"]
        group, split, phase, frame, expected_roi_height = expected_identities[sample_id]
        require((row["group"], row["split"], row["phase"],
                 integer(row["frame_index"], sample_id + " frame_index")) == (group, split, phase, frame),
                "Sample identity mismatch: " + sample_id)
        require(row["view"] == "side", "Expected side view: " + sample_id)
        image_height = integer(row["image_height"], sample_id + " image_height")
        roi_height = integer(row["roi_height"], sample_id + " roi_height")
        peak_row = integer(row["peak_row"], sample_id + " peak_row")
        interface_pixel = number(row["interface_position_pixels"], sample_id + " interface_position_pixels")
        require(image_height == 224, "Unexpected image height: " + sample_id)
        require(roi_height == expected_roi_height, "Unexpected ROI height: " + sample_id)
        require(0 <= peak_row < image_height, "Peak outside attention grid: " + sample_id)
        require(0 <= interface_pixel <= roi_height, "Interface outside ROI: " + sample_id)
        saved_peak = number(row["saved_peak_y_norm"], sample_id + " saved_peak_y_norm")
        saved_interface = number(row["saved_interface_y_norm"], sample_id + " saved_interface_y_norm")
        saved_distance = number(row["saved_distance"], sample_id + " saved_distance")
        require(all(0 <= value <= 1 for value in (saved_peak, saved_interface, saved_distance)),
                "Saved normalized value outside [0, 1]: " + sample_id)
        peak_norm = peak_row / (image_height - 1)
        interface_norm = interface_pixel / roi_height
        close(saved_peak * (image_height - 1), peak_row, sample_id + " recovered peak row")
        close(peak_norm, saved_peak, sample_id + " peak normalization")
        close(interface_norm, saved_interface, sample_id + " interface normalization")
        distance = abs(peak_norm - interface_norm)
        close(distance, saved_distance, sample_id + " saved distance")
        recomputed.append({
            "sample_id": sample_id, "group": group, "split": split, "phase": phase,
            "frame_index": frame, "view": "side", "peak_row": peak_row,
            "image_height": image_height, "interface_position_pixels": interface_pixel,
            "roi_height": roi_height, "peak_y_norm_recomputed": peak_norm,
            "interface_y_norm_recomputed": interface_norm, "distance_recomputed": distance,
            "saved_distance": saved_distance, "absolute_difference": abs(distance - saved_distance),
        })
    source_groups = reference.get("group_means", [])
    require(len(source_groups) == 3, "Expected three reference groups")
    require({row.get("group") for row in source_groups} == {group for group, _, _ in GROUPS},
            "Unexpected reference group identities")
    group_means = []
    for group, split, roi_height in GROUPS:
        saved = next(row for row in source_groups if row["group"] == group)
        require(saved.get("split") == split and saved.get("sample_count") == 3
                and saved.get("original_label_count") == 599 and saved.get("roi_height") == roi_height,
                "Unexpected reference group metadata: " + group)
        values = [row["distance_recomputed"] for row in recomputed if row["group"] == group]
        require(len(values) == 3, "Expected three phases per group: " + group)
        mean = statistics.mean(values)
        close(mean, saved["mean"], group + " mean")
        require(round(mean, 3) == saved["rounded_3dp"], "Rounded group mean mismatch: " + group)
        group_means.append({"group": group, "split": split, "sample_count": len(values),
                            "mean": mean, "reference_mean": saved["mean"],
                            "rounded_3dp": round(mean, 3)})
    overall = statistics.mean(row["distance_recomputed"] for row in recomputed)
    close(overall, reference["overall_mean"], "Overall mean")
    require(round(overall, 3) == reference["overall_rounded_3dp"], "Rounded overall mean mismatch")
    result = {
        "status": "PASS", "module": "attention_distance",
        "verification_scope": "Saved-coordinate normalization and distance arithmetic; no attention-map generation, model inference, or label-provenance verification.",
        "formula": "abs(peak_row / (image_height - 1) - interface_position_pixels / roi_height)",
        "absolute_tolerance": ABS_TOL, "sample_count": len(recomputed),
        "split_counts": split_counts, "selected_frame_indices": [frame for _, frame in PHASES],
        "group_means": group_means, "overall_mean": overall,
        "reference_overall_mean": reference["overall_mean"], "overall_rounded_3dp": round(overall, 3),
        "maximum_difference_from_saved_distance": max(row["absolute_difference"] for row in recomputed),
        "model_rerun": False,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "attention_distance_recomputed_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(recomputed[0]))
        writer.writeheader()
        writer.writerows(recomputed)
    (output_dir / "attention_distance_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR / "verification_output",
                        help="Directory for verification JSON/CSV (default: package/verification_output)")
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
