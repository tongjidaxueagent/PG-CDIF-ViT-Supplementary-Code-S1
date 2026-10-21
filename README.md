# Supplementary Code S1: PG-CDIF-ViT result verification

Verification package version 1.0.0 — prepared 5 October 2026. Repository documentation updated 21 October 2026 for submission to ENGINEERING Environment; scientific inputs, verification scripts and reference results are unchanged.

This repository accompanies the manuscript *A Predictive Model for Settled-Water Turbidity under High-Turbidity Conditions Based on Dual-View Visual Fusion and Interface-Assisted Supervision*. It contains Supplementary Code S1, a selected-result verification package.

## Download and review

Download the repository using **Code → Download ZIP**, extract it, and run the command below from the extracted directory. Keep the `data/`, `config/`, `expected_results/`, and `scripts/` directories together with `run_verification.py`. Use the submission commit or release identified in the manuscript to obtain the reviewed version.

The supplied Python files and numerical inputs retain the original archive's bytes. The Git attributes disable line-ending conversion so the SHA-256 integrity checks also work after a Windows checkout. The checksum manifest includes the updated repository documentation.

This package recalculates selected results from saved mainline turbidity predictions and saved attention/interface coordinates. It supplies independent, small verification scripts and the input records they need. It does not construct a prediction model, regenerate predictions or attention maps, train a model, or provide inference on new images.

## Run

Use Python 3.9 or later. No third-party packages, GPU, network access, pretrained weights, or original project checkout are required.

```text
python run_verification.py
```

Run this command from the directory containing this README, or give the absolute path to `run_verification.py` from another directory. On Windows, `py -3 run_verification.py` is an alternative if Python Launcher is installed.

To keep generated results elsewhere:

```text
python run_verification.py --output-dir verification_run
```

The default output directory is `verification_output/` beside the runner. A successful run prints `PASS` and writes `verification_summary.json` plus the component results. A missing, changed, inconsistent or unsupported input causes a nonzero exit status. `MANIFEST_SHA256.csv` checks packaged file integrity before execution; it is not a digital signature or independent proof of the origin of the scientific inputs.

## Coverage

| Component | Input | What is checked |
|---|---|---|
| Mainline evaluation | `data/mainline_predictions.csv` | MAE, RMSE, implementation-defined MAPE and R² on exported NTU values; original validation/test references and group summaries |
| Experiment-group split | Predictions and `data/group_manifest.csv` | Unique paired-observation identities, expected frame counts, group membership and no group crossing between train/validation/test |
| Attention–interface distance | `data/attention_distance_samples.csv` | Nine side-view representatives, two different normalization denominators, per-sample distances and group/overall arithmetic means |
| Preprocessing parameters | `config/preprocessing_spec.json` | Documented RGB normalization and augmentation settings, one normalization arithmetic example, and the RandAugment operation-selection probability |

The mainline data contain 8,985 paired observations: 7,188 training, 599 validation, and 1,198 test observations. Fifteen experimental groups each contain frame indices 1–599. A row is a paired observation, not a single camera image. Blank split entries in the source export were explicitly assigned to training only after agreement with the independently stored group manifest; the public CSV has no blank splits.

Reference test results from the archived evaluation are MAE 46.5255661011 NTU, RMSE 130.7115783691 NTU, MAPE 19.3365726471%, and R² 0.9249392748. The corresponding common display values are 46.53, 130.71, 19.34%, and 0.9249. The scripts document and check small numerical tolerances because they aggregate saved floating-point values using Python arithmetic rather than replaying the original tensor reductions.

The attention-distance overall mean is approximately 0.1159564713 (0.116 to three decimal places). The low-, medium- and high-initial-turbidity representative group means round to 0.101, 0.072 and 0.175. These nine representatives comprise six test-set frames and three validation-set frames. The overall mean is not a mean over the full test set.

## Interpretation and limits

There are two group-MAPE conventions in the archived material. The current-method group calculation uses `target+1`; archived validation/test group-summary values are separately reproducible with `target` alone. This distinction is preserved and checked, rather than hidden with a wider tolerance. See the explicit formulas and example in `METHODS_AND_DATA.md`. The original overall validation/test MAPE anchors retain the `target+1` convention.

The CSV `target_ntu` values are reference labels exported by the model evaluation. The frame labels are associated with the study's fitted/interpolated turbidity label pipeline; they are not independent instrument measurements at every image frame. This archive does not regenerate or independently validate that label pipeline.

Recalculating metrics from provided predictions checks the statistical calculations. It does not independently establish that a particular model produced those predictions. Checking group manifests does not establish the absence of every possible form of data leakage or prove independence of physical experiments. The distance script checks stored coordinates and aggregation, not the original image registration, label-generation correctness, or model attention computation.

The preprocessing specification reports effective settings inspected in the archived implementation. It is not an executable substitute for the image-processing/training pipeline, and historical per-image random augmentations are not replayed. Only the ROI dimensions required for the selected coordinate checks are included.

Full drawing programs, model architecture source, training/loss implementations, checkpoints, serialized models, raw image collections, other model variants, and software-registration materials are outside this package. This version covers the named mainline results and selected coordinate analysis, not every experiment, figure or table in the article.

See `METHODS_AND_DATA.md` for the formulas and schemas, `PREPROCESSING.md` for S01/S02 details, `PROVENANCE.json` for curation/source identifiers, and `NOTICE.txt` for the limited scope of this prepared release.

## Rights and scope

The existing terms in `NOTICE.txt` are retained. No additional open-source license or rights to the excluded prediction software are granted by this repository. Direct permission or licensing questions to the corresponding author identified in the manuscript.
