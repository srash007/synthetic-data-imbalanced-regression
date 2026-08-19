"""
Benchmark & study script for SERRegressor.
============================================
Matches ser.py's SERRegressor interface:
    SERRegressor(method="A", min_group_n=50, verbose=True, segmenter_kwargs=None)
    .fit(X, y) -> self
    .predict(X) -> np.ndarray
    .evaluate(X_test, y_test) -> dict(global_model, seg, group_models,
                                       group_perf_dfs, pred_naive, pred_blend,
                                       pred_huber, metrics, X_test, y_test)
    .run(X_train, y_train, X_test, y_test) -> shortcut for fit + evaluate
    Post-fit attributes: .method, .seg_, .group_models_, .group_perf_, .alpha_

Part 1 — Quick benchmark: OLS vs SER (single segmentation method),
          full metrics suite + statistical comparison.
Part 2 — SER study: runs all four segmentation methods (A/B/C/D),
          records segmentation thresholds, group sizes, and the
          winning model per group. This is the core contribution
          of the study, so it gets dedicated logging and exports.
"""

import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from ser import SERRegressor
import metrics as mt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



RANDOM_SEED = 42
DATASET_PATH = "datasets/boston.csv"
TARGET_COL = "HousValue"          # explicit target column (CpuSm.csv -> "usr")
SEGMENTATION_METHOD = "D"   # method used in Part 1's single-model benchmark

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_SEED,
)

print(f"Dataset: {DATASET_PATH}  |  Train: {len(X_train)}  |  Test: {len(X_test)}  |  Features: {X.shape[1]}\n")


# ====================================================================
# PART 1 — Quick benchmark: OLS vs SER
# ====================================================================
print("############################################################")
print("PART 1 — OLS vs SER benchmark")
print("############################################################\n")

model_ols = LinearRegression()
model_ser = SERRegressor(SEGMENTATION_METHOD, verbose=False)

start_ols = time.perf_counter()
model_ols.fit(X_train, y_train)
fit_time_ols = time.perf_counter() - start_ols

start_ser = time.perf_counter()
model_ser.fit(X_train, y_train)
fit_time_ser = time.perf_counter() - start_ser

y_pred_ols = np.asarray(model_ols.predict(X_test), dtype=float)
y_pred_ser = np.asarray(model_ser.predict(X_test), dtype=float)
y_true = np.asarray(y_test, dtype=float)

print("=== Fit times (seconds) ===")
print("OLS fit time:", fit_time_ols)
print("SER fit time:", fit_time_ser)
print()

print("=== Regression metrics (OLS) ===")
print("MAE:", mt.mae(y_true, y_pred_ols))
print("MSE:", mt.mse(y_true, y_pred_ols))
print("RMSE:", mt.rmse(y_true, y_pred_ols))
print("R2:", mt.r2(y_true, y_pred_ols))
print("Median AE:", mt.median_absolute_error(y_true, y_pred_ols))
print("MAPE:", mt.mape(y_true, y_pred_ols))
print("sMAPE:", mt.smape(y_true, y_pred_ols))
print("Max Error:", mt.max_error(y_true, y_pred_ols))
print()

print("=== Regression metrics (SER) ===")
print("MAE:", mt.mae(y_true, y_pred_ser))
print("MSE:", mt.mse(y_true, y_pred_ser))
print("RMSE:", mt.rmse(y_true, y_pred_ser))
print("R2:", mt.r2(y_true, y_pred_ser))
print("Median AE:", mt.median_absolute_error(y_true, y_pred_ser))
print("MAPE:", mt.mape(y_true, y_pred_ser))
print("sMAPE:", mt.smape(y_true, y_pred_ser))
print("Max Error:", mt.max_error(y_true, y_pred_ser))
print()

print("=== Imbalance metrics (OLS) ===")
print("Precision phi:", mt.precision_phi(y_true, y_pred_ols))
print("Recall phi:", mt.recall_phi(y_true, y_pred_ols))
print("F1 phi:", mt.f1_phi(y_true, y_pred_ols))
print("Weighted MAE:", mt.weighted_mae(y_true, y_pred_ols))
print("Weighted RMSE:", mt.weighted_rmse(y_true, y_pred_ols))
print("Tail MAE:", mt.tail_mae(y_true, y_pred_ols))
print("Tail RMSE:", mt.tail_rmse(y_true, y_pred_ols))
print("Tail R2:", mt.tail_r2(y_true, y_pred_ols))
print("Tail Coverage:", mt.tail_coverage(y_true, y_pred_ols))
print("Rare Region Error:", mt.rare_region_error(y_true, y_pred_ols))
print()

print("=== Imbalance metrics (SER) ===")
print("Precision phi:", mt.precision_phi(y_true, y_pred_ser))
print("Recall phi:", mt.recall_phi(y_true, y_pred_ser))
print("F1 phi:", mt.f1_phi(y_true, y_pred_ser))
print("Weighted MAE:", mt.weighted_mae(y_true, y_pred_ser))
print("Weighted RMSE:", mt.weighted_rmse(y_true, y_pred_ser))
print("Tail MAE:", mt.tail_mae(y_true, y_pred_ser))
print("Tail RMSE:", mt.tail_rmse(y_true, y_pred_ser))
print("Tail R2:", mt.tail_r2(y_true, y_pred_ser))
print("Tail Coverage:", mt.tail_coverage(y_true, y_pred_ser))
print("Rare Region Error:", mt.rare_region_error(y_true, y_pred_ser))
print()

print("=== Distribution metrics (y_true vs OLS predictions) ===")
print("Wasserstein:", mt.wasserstein_distance(y_true, y_pred_ols))
print("KL divergence:", mt.kl_divergence(y_true, y_pred_ols))
print("Jensen-Shannon:", mt.jensen_shannon_divergence(y_true, y_pred_ols))

print()

print("=== Distribution metrics (y_true vs SER predictions) ===")
print("Wasserstein:", mt.wasserstein_distance(y_true, y_pred_ser))
print("KL divergence:", mt.kl_divergence(y_true, y_pred_ser))
print("Jensen-Shannon:", mt.jensen_shannon_divergence(y_true, y_pred_ser))

print()

# Per-model absolute error vectors for statistical comparisons
err_ols = np.abs(y_true - y_pred_ols)
err_ser = np.abs(y_true - y_pred_ser)
err_dummy = np.abs(y_true - np.mean(y_train))


print("=== High-level evaluation API ===")
results = mt.evaluate_models(
    y_true=y_true,
    predictions={"SER": y_pred_ser, "OLS": y_pred_ols},
)
print(results)

# --------------------------------------------------------------
# Attach experiment metadata — mapped by model name, not by row
# position, so this stays correct regardless of how evaluate_models
# orders its output.
# --------------------------------------------------------------
results = results.reset_index().rename(columns={"index": "model"})

fit_times = {"SER": fit_time_ser, "OLS": fit_time_ols}
segmentation_used = {"SER": model_ser.method, "OLS": np.nan}

results["dataset"] = Path(DATASET_PATH).stem
results["segmentation"] = results["model"].map(segmentation_used)
results["fit_time"] = results["model"].map(fit_times)
results["train_samples"] = len(X_train)
results["test_samples"] = len(X_test)
results["n_features"] = X.shape[1]
results["random_state"] = RANDOM_SEED
results["timestamp"] = pd.Timestamp.now()

csv_path = RESULTS_DIR / "benchmark_results.csv"
if csv_path.exists():
    results.to_csv(csv_path, mode="a", header=False, index=False)
else:
    results.to_csv(csv_path, index=False)

print(f"\nResults saved to {csv_path}")


# ====================================================================
# PART 2 — SER study: compare all segmentation methods (A/B/C/D)
# ====================================================================
print("\n############################################################")
print("PART 2 — SER segmentation study (methods A, B, C, D)")
print("############################################################\n")

segmentation_rows = []
group_perf_rows = []
method_metrics_rows = []

for method in ["A", "B", "C", "D"]:
    print(f"--- Method {method} ---")
    model = SERRegressor(method, verbose=False)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    y_pred = np.asarray(model.predict(X_test), dtype=float)

    # --- segmentation summary ---
    seg = model.seg_
    n_lower = len(seg.idxL)
    n_center = len(seg.idxC)
    n_upper = len(seg.idxU)

    print(f"low_thr={seg.low_thr:.4f}  up_thr={seg.up_thr:.4f}  alpha={model.alpha_:.4f}")
    print(f"group sizes -> Lower={n_lower}  Center={n_center}  Upper={n_upper}")

    for group_name, mdl in model.group_models_.items():
        chosen = getattr(mdl, "name", "None") if mdl is not None else "None (fallback/empty)"
        print(f"  {group_name}: best model = {chosen}")

    segmentation_rows.append({
        "method": method,
        "low_thr": seg.low_thr,
        "up_thr": seg.up_thr,
        "alpha": model.alpha_,
        "n_lower": n_lower,
        "n_center": n_center,
        "n_upper": n_upper,
        "lower_model": getattr(model.group_models_.get("Lower"), "name", None),
        "center_model": getattr(model.group_models_.get("Center"), "name", None),
        "upper_model": getattr(model.group_models_.get("Upper"), "name", None),
        "fit_time": fit_time,
    })

    # --- per-group candidate performance (all models tried per group) ---
    for group_name, perf_df in model.group_perf_.items():
        if perf_df is not None and not perf_df.empty:
            tmp = perf_df.copy()
            tmp["method"] = method
            tmp["group"] = group_name
            group_perf_rows.append(tmp[["method", "group", "name", "R2", "MAE", "RMSE"]])

    # --- test-set metrics for this segmentation method ---
    y_true_arr = np.asarray(y_test, dtype=float)
    method_metrics_rows.append({
        "method": method,
        "R2": mt.r2(y_true_arr, y_pred),
        "MAE": mt.mae(y_true_arr, y_pred),
        "RMSE": mt.rmse(y_true_arr, y_pred),
        "Tail_MAE": mt.tail_mae(y_true_arr, y_pred),
        "Tail_R2": mt.tail_r2(y_true_arr, y_pred),
        "F1_phi": mt.f1_phi(y_true_arr, y_pred),
        "fit_time": fit_time,
    })
    print()

segmentation_df = pd.DataFrame(segmentation_rows)
group_perf_df = pd.concat(group_perf_rows, ignore_index=True) if group_perf_rows else pd.DataFrame()
method_metrics_df = pd.DataFrame(method_metrics_rows)

print("=== Segmentation summary (all methods) ===")
print(segmentation_df)
print()
print("=== Test-set metrics per segmentation method ===")
print(method_metrics_df.sort_values("R2", ascending=False))

# --------------------------------------------------------------
# Export the full SER study to a single Excel workbook
# --------------------------------------------------------------
study_path = RESULTS_DIR / f"ser_study_{Path(DATASET_PATH).stem}.xlsx"
with pd.ExcelWriter(study_path) as writer:
    segmentation_df.to_excel(writer, sheet_name="Segmentation_Summary", index=False)
    method_metrics_df.to_excel(writer, sheet_name="Method_Metrics", index=False)
    if not group_perf_df.empty:
        group_perf_df.to_excel(writer, sheet_name="Group_Candidate_Perf", index=False)

print(f"\nSER study saved to {study_path}")