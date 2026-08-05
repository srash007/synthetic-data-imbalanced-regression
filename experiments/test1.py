import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ser import SERRegressor
import metrics as mt

RANDOM_SEED = 42

# Load dataset
path = "datasets/CpuSm.csv"
df = pd.read_csv(path)

# Use numeric-only features for a straightforward test script
X = df.select_dtypes(include=["number"]).iloc[:, :-1]
y = df.select_dtypes(include=["number"]).iloc[:, -1]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_SEED,
)

# Train models
model_ols = LinearRegression()
model_ser = SERRegressor("D")

start_ols = time.perf_counter()
model_ols.fit(X_train, y_train)
fit_time_ols = time.perf_counter() - start_ols

start_ser = time.perf_counter()
model_ser.fit(X_train, y_train)
fit_time_ser = time.perf_counter() - start_ser

# Predict
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

print("=== Distribution metrics (y_true vs OLS predictions) ===")
print("Wasserstein:", mt.wasserstein_distance(y_true, y_pred_ols))
print("KL divergence:", mt.kl_divergence(y_true, y_pred_ols))
print("Jensen-Shannon:", mt.jensen_shannon_divergence(y_true, y_pred_ols))
print("Kolmogorov-Smirnov:", mt.kolmogorov_smirnov(y_true, y_pred_ols))
print("Cramer-von-Mises:", mt.cramer_von_mises(y_true, y_pred_ols))
print("Energy distance:", mt.energy_distance(y_true, y_pred_ols))
print()

# Build per-model absolute error vectors for statistical comparisons
err_ols = np.abs(y_true - y_pred_ols)
err_ser = np.abs(y_true - y_pred_ser)
err_dummy = np.abs(y_true - np.mean(y_train))

print("=== Statistical utilities (errors comparison) ===")
print("Paired t-test:", mt.paired_ttest(err_ols, err_ser))
print("Wilcoxon test:", mt.wilcoxon_test(err_ols, err_ser))
print("Friedman test:", mt.friedman_test(err_ols, err_ser, err_dummy))
print("Nemenyi test:", mt.nemenyi_test(err_ols, err_ser, err_dummy))
print(
    "Bootstrap CI:",
    mt.bootstrap_confidence_interval(err_ols - err_ser, statistic="mean", n_bootstrap=500),
)
print("Cohen's d:", mt.cohens_d(err_ols, err_ser))
print("Cliff's delta:", mt.cliffs_delta(err_ols, err_ser))
print()

print("=== High-level evaluation API ===")
results = mt.evaluate_models(
    y_true=y_true,
    predictions={
        "SER": y_pred_ser,
        "OLS": y_pred_ols,
    },
)
print(results)

# ------------------------------------------------------------------
# Add experiment metadata
# ------------------------------------------------------------------
results = results.reset_index().rename(columns={"index": "model"})

results["dataset"] = Path(path).stem
results["segmentation"] = ["A", np.nan]
results["fit_time"] = [fit_time_ser, fit_time_ols]
results["train_samples"] = len(X_train)
results["test_samples"] = len(X_test)
results["n_features"] = X.shape[1]
results["random_state"] = 42
results["timestamp"] = pd.Timestamp.now() 

# ------------------------------------------------------------------
# Save results
# ------------------------------------------------------------------
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

csv_path = results_dir / "benchmark_results.csv"

if csv_path.exists():
    results.to_csv(csv_path, mode="a", header=False, index=False)
else:
    results.to_csv(csv_path, index=False)

print(f"\nResults saved to {csv_path}")