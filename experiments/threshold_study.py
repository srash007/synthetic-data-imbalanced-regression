"""
Threshold comparison study: SER vs SMOGN+OLS vs plain OLS.
================================================================
Runs SER *and* SMOGN+OLS across several DIFFERENT region-definition
thresholds -- plain quantile cuts and/or SMOGN-relevance-derived cuts
-- using the SAME train/test split AND the SAME RegionDefinition for
both methods at each threshold config, so the comparison is fair.

For each threshold config, records:
  - a segmentation summary (thresholds, group sizes, winning SER model
    per group)
  - the full candidate-model performance table per SER group
  - test-set metrics for SER, SMOGN+OLS, and plain OLS (one row each)

Results print to console and are saved to CSV and/or Excel.

How to run
----------
Place this file in your `experiments/` folder (next to your other
scripts) and run from the project root:

    python experiments/threshold_study.py

Configure DATASET_PATH, TARGET_COL, THRESHOLD_CONFIGS, SMOGN_* and
EXPORT_FORMAT below.
"""

import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ser import SERRegressor
from segmentation.regions import RegionDefinition
from segmentation.smogn_relevance import extract_smogn_thresholds
from preprocessing.smogn import SMOGNPreprocessor
import metrics as mt


# ====================================================================
# Configuration
# ====================================================================
RANDOM_SEED = 42
DATASET_PATH = "datasets/bank8fm.csv"
TARGET_COL = "rej"
TEST_SIZE = 0.30
EXPORT_FORMAT = "xlsx"   # "xlsx", "csv", or "both"

# SMOGN resampling parameters, applied identically at every threshold
# config (only the region boundary changes between configs).
SMOGN_REL_THRES = 0.80   # relevance threshold used to drive bump-splitting
                          # off the manual control points anchored at each
                          # config's low_thr/up_thr -- see preprocessing/smogn.py
SMOGN_K = 5
SMOGN_PERT = 0.02
SMOGN_SAMP_METHOD = "balance"

# Fixed quantiles used ONLY to stratify the train/test split (keeps the
# same tail proportions in train/test regardless of which threshold
# config is being evaluated). This is independent of the per-config
# thresholds tested below.
SPLIT_Q_LOW = 0.15
SPLIT_Q_UP = 0.85

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Each entry is one threshold configuration to compare. "kind" controls
# how the low/up thresholds are computed:
#   "quantile" -> fixed q_low / q_up quantiles of y_train
#   "smogn"    -> derived from SMOGN's own relevance function (rel_thres)
#   "fixed"    -> literal low/up values you specify directly
THRESHOLD_CONFIGS = [
    {"name": "quantile_10_90", "kind": "quantile", "q_low": 0.10, "q_up": 0.90},
    {"name": "quantile_15_85", "kind": "quantile", "q_low": 0.15, "q_up": 0.85},
    {"name": "quantile_20_80", "kind": "quantile", "q_low": 0.20, "q_up": 0.80},
    {"name": "smogn_rel_0.70", "kind": "smogn", "rel_thres": 0.70},
    {"name": "smogn_rel_0.80", "kind": "smogn", "rel_thres": 0.80},
    {"name": "smogn_rel_0.90", "kind": "smogn", "rel_thres": 0.90},
    # Example of a manually fixed pair -- edit or remove as needed:
    # {"name": "fixed_14_32", "kind": "fixed", "low": 14.09, "up": 32.01},
]


# ====================================================================
# Helpers
# ====================================================================
def build_region(config, y_train):
    """Build a RegionDefinition from one entry of THRESHOLD_CONFIGS."""
    kind = config["kind"]
    y_arr = np.asarray(y_train, dtype=float)

    if kind == "quantile":
        low_thr = float(np.quantile(y_arr, config["q_low"]))
        up_thr = float(np.quantile(y_arr, config["q_up"]))
        return RegionDefinition.from_thresholds(y=y_arr, lower=low_thr, upper=up_thr)

    if kind == "smogn":
        thresholds = extract_smogn_thresholds(
            y=y_arr,
            rel_thres=config["rel_thres"],
            rel_xtrm_type=config.get("rel_xtrm_type", "both"),
            rel_coef=config.get("rel_coef", 1.5),
        )
        low_thr, up_thr = thresholds["low_thr"], thresholds["up_thr"]

        # SMOGN found no rare observations on one side (e.g. a right-skewed
        # target with no low-tail outliers under the box-plot rule). Rather
        # than discard the whole config, pin the missing bound to the
        # train set's own min/max: the corresponding group simply ends up
        # empty, and Center absorbs everything else.
        if low_thr is None:
            low_thr = float(np.min(y_arr))
        if up_thr is None:
            up_thr = float(np.max(y_arr))

        return RegionDefinition.from_thresholds(y=y_arr, lower=low_thr, upper=up_thr)

    if kind == "fixed":
        return RegionDefinition.from_thresholds(y=y_arr, lower=config["low"], upper=config["up"])

    raise ValueError(f"Unknown threshold kind: {kind!r} (config name: {config.get('name')})")


def build_config_metrics(config_name, kind, y_true, predictions: dict, fit_times: dict):
    """
    Runs metrics.evaluate_models() (the full regression + imbalance metric
    registry) over every model's predictions for one threshold config, and
    tags the result with config/kind/fit_time so configs can be
    concatenated into one long DataFrame afterward.
    """
    metrics_df = mt.evaluate_models(y_true=y_true, predictions=predictions)
    metrics_df = metrics_df.reset_index().rename(columns={"index": "model"})
    metrics_df.insert(0, "config", config_name)
    metrics_df.insert(1, "kind", kind)
    metrics_df["fit_time"] = metrics_df["model"].map(fit_times)
    return metrics_df


def save_results(dataframes: dict, export_format: str, out_stem: str):
    """Save {sheet_name: DataFrame} either as one Excel workbook (multiple
    sheets) and/or as separate CSV files, per EXPORT_FORMAT. Falls back to
    CSV automatically if openpyxl isn't installed, instead of crashing."""
    export_format = export_format.lower()
    saved_paths = []

    want_xlsx = export_format in ("xlsx", "both")
    want_csv = export_format in ("csv", "both")

    if want_xlsx:
        try:
            xlsx_path = RESULTS_DIR / f"{out_stem}.xlsx"
            with pd.ExcelWriter(xlsx_path) as writer:
                for sheet_name, df in dataframes.items():
                    if df is not None and not df.empty:
                        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            saved_paths.append(xlsx_path)
        except ImportError:
            print(
                "openpyxl not installed (`pip install openpyxl`) -- "
                "falling back to CSV export instead of failing."
            )
            want_csv = True

    if want_csv:
        for sheet_name, df in dataframes.items():
            if df is not None and not df.empty:
                csv_path = RESULTS_DIR / f"{out_stem}_{sheet_name}.csv"
                df.to_csv(csv_path, index=False)
                saved_paths.append(csv_path)

    return saved_paths


# ====================================================================
# Load data
# ====================================================================
df = pd.read_csv(DATASET_PATH)

if TARGET_COL not in df.columns:
    raise KeyError(
        f"Target column '{TARGET_COL}' not found in {DATASET_PATH}. "
        f"Available columns: {list(df.columns)}"
    )

X = df.drop(columns=[TARGET_COL]).select_dtypes(include=["number"])
y = df[TARGET_COL]

# ====================================================================
# Stratified train/test split (shared across every threshold config
# AND every model -- SER, SMOGN+OLS, plain OLS all see the same split)
# ====================================================================
split_low = float(y.quantile(SPLIT_Q_LOW))
split_up = float(y.quantile(SPLIT_Q_UP))
split_labels = np.where(y < split_low, "Lower", np.where(y > split_up, "Upper", "Center"))

train_idx, test_idx = train_test_split(
    np.arange(len(y)), test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=split_labels,
)

X_train = X.iloc[train_idx].reset_index(drop=True)
y_train = y.iloc[train_idx].reset_index(drop=True)
X_test = X.iloc[test_idx].reset_index(drop=True)
y_test = y.iloc[test_idx].reset_index(drop=True)

print(f"Dataset: {DATASET_PATH}  |  Train: {len(X_train)}  |  Test: {len(X_test)}  |  Features: {X.shape[1]}")
print(f"Threshold configs to compare: {[c['name'] for c in THRESHOLD_CONFIGS]}\n")

# Plain OLS baseline: doesn't depend on any region/threshold, computed
# once and reused as the reference row for every config.
t0 = time.perf_counter()
ols_plain = LinearRegression().fit(X_train, y_train)
fit_time_ols_plain = time.perf_counter() - t0
y_pred_ols_plain = ols_plain.predict(X_test)


# ====================================================================
# Sweep across threshold configurations
# ====================================================================
segmentation_rows = []
group_perf_rows = []
method_metrics_rows = []

for config in THRESHOLD_CONFIGS:
    name = config["name"]
    kind = config["kind"]
    print(f"--- {name} ({kind}) ---")

    try:
        region = build_region(config, y_train)
    except Exception as e:
        print(f"  SKIPPED -- could not build region: {e}\n")
        continue

    # ---------------- SER, trained on this config's region ----------------
    ser_model = SERRegressor(method="B", verbose=False)  # method irrelevant once region_definition is passed

    t0 = time.perf_counter()
    ser_model.fit(X_train, y_train, region_definition=region)
    fit_time_ser = time.perf_counter() - t0

    y_pred_ser = np.asarray(ser_model.predict(X_test), dtype=float)

    n_lower = int(region.lower_mask.sum())
    n_center = int(region.center_mask.sum())
    n_upper = int(region.upper_mask.sum())

    print(f"  low_thr={ser_model.low_thr_:.4f}  up_thr={ser_model.up_thr_:.4f}  alpha={ser_model.alpha_:.4f}")
    print(f"  group sizes -> Lower={n_lower}  Center={n_center}  Upper={n_upper}")
    for group_name, mdl in ser_model.group_models_.items():
        chosen = getattr(mdl, "name", "None") if mdl is not None else "None (fallback/empty)"
        print(f"    {group_name}: best model = {chosen}")

    segmentation_rows.append({
        "config": name,
        "kind": kind,
        "low_thr": ser_model.low_thr_,
        "up_thr": ser_model.up_thr_,
        "alpha": ser_model.alpha_,
        "n_lower": n_lower,
        "n_center": n_center,
        "n_upper": n_upper,
        "lower_model": getattr(ser_model.group_models_.get("Lower"), "name", None),
        "center_model": getattr(ser_model.group_models_.get("Center"), "name", None),
        "upper_model": getattr(ser_model.group_models_.get("Upper"), "name", None),
        "fit_time": fit_time_ser,
    })

    for group_name, perf_df in ser_model.group_perf_.items():
        if perf_df is not None and not perf_df.empty:
            tmp = perf_df.copy()
            tmp["config"] = name
            tmp["group"] = group_name
            group_perf_rows.append(tmp[["config", "group", "name", "R2", "MAE", "RMSE"]])

    # ---------------- SMOGN + OLS, on the SAME region ----------------
    predictions = {"SER": y_pred_ser}
    fit_times = {"SER": fit_time_ser}

    try:
        smogn_pre = SMOGNPreprocessor(
            rel_thres=SMOGN_REL_THRES, k=SMOGN_K, pert=SMOGN_PERT, samp_method=SMOGN_SAMP_METHOD,
        )
        t0 = time.perf_counter()
        X_res, y_res = smogn_pre.fit_resample(X_train, y_train, region_definition=region)
        ols_smogn = LinearRegression().fit(X_res, y_res)
        fit_time_smogn = time.perf_counter() - t0

        y_pred_smogn = ols_smogn.predict(X_test)
        print(f"  SMOGN+OLS: resampled train n={len(y_res)} (original {len(y_train)})")

        predictions["SMOGN_OLS"] = y_pred_smogn
        fit_times["SMOGN_OLS"] = fit_time_smogn
    except Exception as e:
        print(f"  SMOGN+OLS FAILED for this config: {e}")
        print(
            "  (if this is a numpy>=2.0 TypeError inside over_sampling, "
            "pin numpy: pip install \"numpy<2\")"
        )

    # ---------------- Plain OLS baseline, same prediction for every config ----------------
    predictions["OLS"] = y_pred_ols_plain
    fit_times["OLS"] = fit_time_ols_plain

    # One evaluate_models() call per config -> full metric registry
    # (mae, mse, rmse, r2, median_absolute_error, mape, smape, max_error,
    # precision_phi, recall_phi, f1_phi, weighted_mae, weighted_rmse,
    # tail_mae, tail_rmse, tail_r2, tail_coverage, rare_region_error) for
    # every model that produced predictions at this threshold config.
    method_metrics_rows.append(
        build_config_metrics(name, kind, y_test, predictions, fit_times)
    )

    print()

segmentation_df = pd.DataFrame(segmentation_rows)
group_perf_df = pd.concat(group_perf_rows, ignore_index=True) if group_perf_rows else pd.DataFrame()
method_metrics_df = (
    pd.concat(method_metrics_rows, ignore_index=True) if method_metrics_rows else pd.DataFrame()
)

print("=== Segmentation summary (SER, all threshold configs) ===")
print(segmentation_df)
print()
print("=== Test-set metrics: SER vs SMOGN+OLS vs OLS, per threshold config ===")
print(method_metrics_df.sort_values(["config", "r2"], ascending=[True, False]))

# ====================================================================
# Save results
# ====================================================================
out_stem = f"threshold_study_{Path(DATASET_PATH).stem}"
saved_paths = save_results(
    dataframes={
        "Segmentation_Summary": segmentation_df,
        "Method_Metrics": method_metrics_df,
        "Group_Candidate_Perf": group_perf_df,
    },
    export_format=EXPORT_FORMAT,
    out_stem=out_stem,
)

print("\nSaved:")
for p in saved_paths:
    print(f"  {p}")