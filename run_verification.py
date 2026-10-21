"""Run the selected verification tasks using Python's standard library only.

The package consumes saved predictions and coordinates. It neither constructs
a prediction model nor loads trained weights. Run with Python 3.9 or later.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys


BASE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def verify_manifest():
    """核对发布清单中的相对路径、文件大小与 SHA256。"""
    manifest = BASE / "MANIFEST_SHA256.csv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "bytes", "sha256"]:
            raise ValueError("Invalid manifest columns")
        records = list(reader)
    seen = set()
    for record in records:
        relative = record["path"]
        target = (BASE / relative).resolve()
        if not relative or relative in seen or target == manifest.resolve():
            raise ValueError("Empty, duplicate, or self-referencing manifest entry")
        if BASE not in target.parents:
            raise ValueError("Manifest path escapes the package")
        if not target.is_file():
            raise ValueError("Missing packaged file: " + relative)
        content = target.read_bytes()
        if len(content) != int(record["bytes"]):
            raise ValueError("File size mismatch: " + relative)
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError("SHA256 mismatch: " + relative)
        seen.add(relative)
    required = {
        "run_verification.py", "scripts/verify_metrics.py",
        "scripts/check_split_manifest.py", "scripts/verify_attention_distance.py",
        "data/mainline_predictions.csv", "data/group_manifest.csv",
        "data/attention_distance_samples.csv", "config/preprocessing_spec.json",
        "expected_results/metrics_reference.json", "expected_results/split_reference.json",
        "expected_results/attention_distance_reference.json",
    }
    if not required.issubset(seen):
        raise ValueError("Required files missing from manifest")
    return {"status": "PASS", "files_checked": len(seen),
            "purpose": "File integrity, not proof of the scientific origin of input data."}


def check_preprocessing_spec(output_dir):
    """检查参数说明中的归一化算例与增强选择概率，不执行图像增强。"""
    spec = json.loads((BASE / "config/preprocessing_spec.json").read_text(encoding="utf-8"))
    norm = spec["normalization"]
    if norm["channel_order"] != ["R", "G", "B"]:
        raise ValueError("Unexpected normalization channel order")
    if norm["mean"] != [0.485, 0.456, 0.406] or norm["std"] != [0.229, 0.224, 0.225]:
        raise ValueError("Normalization constants do not match the archived method")
    pixel = [0, 128, 255]
    normalized = [(x / norm["pixel_divisor"] - mu) / sigma
                  for x, mu, sigma in zip(pixel, norm["mean"], norm["std"])]
    reference = [-2.1179039301310043, 0.20518207282913153, 2.64]
    if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
           for a, b in zip(normalized, reference)):
        raise ValueError("Normalization example failed")
    ra = spec["training_augmentation"]["rand_augment"]
    pool_size = len(ra["operation_pool"])
    probability = 1 - ((pool_size - 1) / pool_size) ** ra["num_ops"]
    if pool_size != 14 or ra["num_ops"] != 2 or ra["magnitude_index"] != 9:
        raise ValueError("Unexpected RandAugment configuration")
    if not math.isclose(probability, 27 / 196, rel_tol=0, abs_tol=1e-15):
        raise ValueError("RandAugment selection-probability check failed")
    result = {
        "status": "PASS", "input_rgb_uint8": pixel,
        "normalized_rgb_float64": normalized,
        "brightness_or_contrast_each_probability_at_least_once": probability,
        "scope": "Parameter arithmetic only; no raw image processing or historical random draws are replayed.",
    }
    (output_dir / "preprocessing_parameter_check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def execute_module(filename, output_dir):
    """从已通过完整性检查的包内脚本调用独立核验入口。"""
    path = BASE / "scripts" / filename
    module_spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.verify(output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE / "verification_output")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir == BASE or output_dir in {BASE / "data", BASE / "config", BASE / "expected_results", BASE / "scripts"}:
        parser.error("Choose a separate output directory, not a package input directory")
    report = {"package_version": "1.0.0", "python_version": platform.python_version(),
              "status": "FAIL", "checks": {}}
    try:
        report["checks"]["file_integrity"] = verify_manifest()
        sys.path.insert(0, str(BASE / "scripts"))
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, filename in [
            ("split", "check_split_manifest.py"),
            ("metrics", "verify_metrics.py"),
            ("attention_distance", "verify_attention_distance.py"),
        ]:
            result = execute_module(filename, output_dir)
            if str(result.get("status", "")).lower() not in {"pass", "passed"}:
                raise ValueError("Verification component did not report success: " + name)
            report["checks"][name] = result
            print("PASS: " + name)
        report["checks"]["preprocessing_parameters"] = check_preprocessing_spec(output_dir)
        print("PASS: preprocessing parameter arithmetic")
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = str(error)
        print("FAIL: " + str(error), file=sys.stderr)
        if output_dir.is_dir():
            (output_dir / "verification_summary.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 1
    (output_dir / "verification_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PASS: all selected checks. Results: " + str(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
