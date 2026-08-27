"""
Does SER outperform synthetic oversampling methods?
================================================================
Compares SER against four resampling-based baselines, all paired with
plain OLS, all sharing the SAME train/test split AND the SAME
RegionDefinition (rare/normal boundary):

    SER
    SMOGN        + OLS
    SmoteR       + OLS
    Gaussian Noise resampling + OLS
    Random UnderSampling      + OLS
    OLS (no resampling, reference baseline)

Every resampling method targets the same balanced composition
(samp_method="balance"); only the mechanism used to add/remove points
differs. This isolates the actual research question: does SER's
region-aware modeling beat data-level rebalancing, method for method?

How to run
----------
    python experiments/q1_resampling_comparison.py

Configure DATASET_PATH, TARGET_COL, REGION_CONFIG, and EXPORT_FORMAT
below.
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
from preprocessing.resamplers import (
    SmoteRPreprocessor,
    GaussianNoisePreprocessor,
    RandomUnderSamplingPreprocessor,
)
import metrics as mt


# ====================================================================
# Configuration
# ====================================================================
RANDOM_SEED = 42
DATASET_PATH = "datasets/CpuSm.csv"
TARGET_COL = "usr"
TEST_SIZE = 0.30
EXPORT_FORMAT = "xlsx"   # "xlsx", "csv", or "both"

# Single shared region definition used by every method in this study.
# "kind": "quantile" (q_low/q_up) or "smogn" (rel_thres-derived).
REGION_CONFIG = {"kind": "quantile", "q_low": 0.15, "q_up": 0.85}

# Shared resampling parameters -- same balance target and neighbor/
# perturbation settings across SMOGN, SmoteR, and Gaussian Noise, so
# differences in results trace back to the generation mechanism, not to
# mismatched hyperparameters.
SAMP_METHOD = "balance"
K_NEIGHBORS = 5
PERT = 0.02
SMOGN_REL_THRES = 0.80  # only used to drive SMOGN's internal bump split

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ====================================================================
# Helpers
# ====================================================================
def build_region(config, y_train):
    y_arr = np.asarray(y_train, dtype=float)
    kind = config["kind"]

    if kind == "quantile":
        low_thr = float(np.quantile(y_arr, config["q_low"]))
        up_thr = float(np.quantile(y_arr, config["q_up"]))
        return RegionDefinition.from_thresholds(y=y_arr, lower=low_thr, upper=up_thr)

    if kind == "smogn":
        thresholds = extract_smogn_thresholds(
            y=y_arr, rel_thres=config["rel_thres"],
            rel_xtrm_type=config.get("rel_xtrm_type", "both"),
            rel_coef=config.get("rel_coef", 1.5),
        )
        low_thr = thresholds["low_thr"] if thresholds["low_thr"] is not None else float(np.min(y_arr))
        up_thr = thresholds["up_thr"] if thresholds["up_thr"] is not None else float(np.max(y_arr))
        return RegionDefinition.from_thresholds(y=y_arr, lower=low_thr, upper=up_thr)

    raise ValueError(f"Unknown region kind: {kind!r}")


def save_results(dataframes: dict, export_format: str, out_stem: str):
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
            print("openpyxl not installed (`pip install openpyxl`) -- falling back to CSV.")
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
    raise KeyError(f"Target column '{TARGET_COL}' not found. Available: {list(df.columns)}")

X = df.drop(columns=[TARGET_COL]).select_dtypes(include=["number"])
y = df[TARGET_COL]

# ====================================================================
# Stratified train/test split (shared by every method)
# ====================================================================
split_low = float(y.quantile(0.15))
split_up = float(y.quantile(0.85))
split_labels = np.where(y < split_low, "Lower", np.where(y > split_up, "Upper", "Center"))

train_idx, test_idx = train_test_split(
    np.arange(len(y)), test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=split_labels,
)
X_train = X.iloc[train_idx].reset_index(drop=True)
y_train = y.iloc[train_idx].reset_index(drop=True)
X_test = X.iloc[test_idx].reset_index(drop=True)
y_test = y.iloc[test_idx].reset_index(drop=True)

print(f"Dataset: {DATASET_PATH}  |  Train: {len(X_train)}  |  Test: {len(X_test)}  |  Features: {X.shape[1]}")

# ====================================================================
# Shared RegionDefinition for every method
# ====================================================================
region = build_region(REGION_CONFIG, y_train)
n_rare = int(region.rare_mask.sum())
n_normal = int(region.normal_mask.sum())
print(f"Shared region: lower={region.lower_threshold:.4f}  upper={region.upper_threshold:.4f}")
print(f"Rare={n_rare}  Normal={n_normal}\n")


# ====================================================================
# Train every method
# ====================================================================
predictions = {}
fit_times = {}
resample_sizes = {}

# ---- SER ----
print("--- SER ---")
t0 = time.perf_counter()
ser_model = SERRegressor(method="B", verbose=False)
ser_model.fit(X_train, y_train, region_definition=region)
fit_times["SER"] = time.perf_counter() - t0
predictions["SER"] = np.asarray(ser_model.predict(X_test), dtype=float)
print(f"  fit_time={fit_times['SER']:.2f}s")

# ---- OLS (no resampling, reference) ----
print("--- OLS (no resampling) ---")
t0 = time.perf_counter()
ols_plain = LinearRegression().fit(X_train, y_train)
fit_times["OLS"] = time.perf_counter() - t0
predictions["OLS"] = ols_plain.predict(X_test)
print(f"  fit_time={fit_times['OLS']:.2f}s")

# ---- Resampling methods, each paired with OLS ----
resamplers = {
    "SMOGN_OLS": SMOGNPreprocessor(
        rel_thres=SMOGN_REL_THRES, k=K_NEIGHBORS, pert=PERT, samp_method=SAMP_METHOD,
    ),
    "SmoteR_OLS": SmoteRPreprocessor(
        k=K_NEIGHBORS, samp_method=SAMP_METHOD, random_state=RANDOM_SEED,
    ),
    "GaussianNoise_OLS": GaussianNoisePreprocessor(
        pert=PERT, samp_method=SAMP_METHOD, random_state=RANDOM_SEED,
    ),
    "RandomUnderSampling_OLS": RandomUnderSamplingPreprocessor(
        samp_method=SAMP_METHOD, random_state=RANDOM_SEED,
    ),
}

for name, resampler in resamplers.items():
    print(f"--- {name} ---")
    try:
        t0 = time.perf_counter()
        X_res, y_res = resampler.fit_resample(X_train, y_train, region_definition=region)
        model = LinearRegression().fit(X_res, y_res)
        fit_times[name] = time.perf_counter() - t0

        predictions[name] = model.predict(X_test)
        resample_sizes[name] = len(y_res)
        print(f"  resampled train n={len(y_res)} (original {len(y_train)})  fit_time={fit_times[name]:.2f}s")
    except Exception as e:
        print(f"  FAILED: {e}")
        if "over_sampling" in str(e) or "0-dimensional" in str(e):
            print('  (numpy>=2.0 incompatibility in the smogn package -- pin numpy: pip install "numpy<2")')

print()


# ====================================================================
# Evaluate all methods with the full metric registry
# ====================================================================
metrics_df = mt.evaluate_models(y_true=y_test, predictions=predictions)
metrics_df = metrics_df.reset_index().rename(columns={"index": "model"})
metrics_df["fit_time"] = metrics_df["model"].map(fit_times)
metrics_df["resampled_train_n"] = metrics_df["model"].map(resample_sizes)

print("=== results: SER vs resampling baselines ===")
print(metrics_df.sort_values("r2", ascending=False).to_string(index=False))


# ====================================================================
# Save
# ====================================================================
out_stem = f"resampling_comparison_{Path(DATASET_PATH).stem}"
saved_paths = save_results(
    dataframes={"Metrics": metrics_df},
    export_format=EXPORT_FORMAT,
    out_stem=out_stem,
)

print("\nSaved:")
for p in saved_paths:
    print(f"  {p}")