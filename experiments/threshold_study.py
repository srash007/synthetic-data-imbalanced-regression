"""
Compare SMOGN + OLS versus SER using the SAME target segmentation.

Fair comparison: both methods use exactly the same RegionDefinition
computed once from the training set.

Author
------
Sarah Elyane Rashiwa
"""
from __future__ import annotations
import time
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from ser import SERRegressor
from preprocessing.smogn import SMOGNPreprocessor
from segmentation.regions import RegionDefinition
from ser import SERRegressor
import metrics as mt



RANDOM_SEED = 42
DATASET_PATH = "datasets/boston.csv"
TARGET_COL = "HousValue"          # explicit target column (CpuSm.csv -> "usr")
SEGMENTATION_METHOD = "B"   # method used in Part 1's single-model benchmark

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

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print(f"Dataset: {DATASET_PATH}  |  Train: {len(X_train)}  |  Test: {len(X_test)}  |  Features: {X.shape[1]}\n")


# =============================================================================
# 3. Build ONE RegionDefinition from y_train
#    (quantile-based for reproducibility)
# =============================================================================

q_low = 0.15
q_up = 0.85

lower_threshold = float(y_train.quantile(q_low))
upper_threshold = float(y_train.quantile(q_up))

regions = RegionDefinition.from_thresholds(
    y=y_train.to_numpy(),
    lower=lower_threshold,
    upper=upper_threshold,
)

print("\n" + "=" * 50)
print("Shared Region Definition")
print("=" * 50)
regions.summary()

# =============================================================================
# 4. Train SER with the shared RegionDefinition
# =============================================================================

print("\n" + "=" * 50)
print("SER Training")
print("=" * 50)

ser_model = SERRegressor(
    method="B",        # use quantile-based (will be overridden by region_definition)
    verbose=True,
)

ser_model.fit(
    X_train,
    y_train,
    region_definition=regions,   # <-- shared object
)

y_pred_ser = ser_model.predict(X_test)

# =============================================================================
# 5. Apply SMOGN with the SAME RegionDefinition
# =============================================================================

print("\n" + "=" * 50)
print("SMOGN Resampling (with shared RegionDefinition)")
print("=" * 50)

smogn_preprocessor = SMOGNPreprocessor(
    rel_thres=0.8,
    rel_method="auto",
    samp_method="balance",
    k=5,
    pert=0.02,
)

X_resampled, y_resampled = smogn_preprocessor.fit_resample(
    X_train,
    y_train,
    region_definition=regions,   # <-- same shared object
)

print(f"Original train size   : {len(y_train)}")
print(f"Resampled train size  : {len(y_resampled)}")

# =============================================================================
# 6. Train OLS on SMOGN-resampled data
# =============================================================================

print("\n" + "=" * 50)
print("OLS on SMOGN-resampled data")
print("=" * 50)

ols_model = LinearRegression()
ols_model.fit(X_resampled, y_resampled)

y_pred_ols_smogn = ols_model.predict(X_test)

y_pred_ols_plain = LinearRegression().fit(X_train, y_train).predict(X_test)

# =============================================================================
# 7. Evaluation
# =============================================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

results = mt.evaluate_models(
    y_true=y_test,
    predictions={
        "SER": y_pred_ser,
        "SMOGN + OLS": y_pred_ols_smogn,
        "OLS": y_pred_ols_plain,
    },
)

print(results)

results.to_csv(
    RESULTS_DIR / "benchmark_results.csv",
    index=True,
)

# =============================================================================
# 8. Distribution metrics
# =============================================================================

distribution_results = pd.DataFrame(
    {
        "Wasserstein": [
            mt.wasserstein_distance(
                y_train,
                y_resampled,
            )
        ],
        "KL Divergence": [
            mt.kl_divergence(
                y_train,
                y_resampled,
            )
        ],
        "Jensen-Shannon": [
            mt.jensen_shannon_divergence(
                y_train,
                y_resampled,
            )
        ],
        "Kolmogorov-Smirnov": [
            mt.kolmogorov_smirnov(
                y_train,
                y_resampled,
            )
        ],
        "Energy Distance": [
            mt.energy_distance(
                y_train,
                y_resampled,
            )
        ],
    },
    index=["SMOGN"],
)

print("\n")
print("=" * 70)
print("DISTRIBUTION METRICS")
print("=" * 70)
print(distribution_results)

distribution_results.to_csv(
    RESULTS_DIR / "distribution_metrics.csv"
)

# =============================================================================
# 9. Statistical comparison
# =============================================================================

ser_errors = np.abs(y_test - y_pred_ser)
smogn_errors = np.abs(y_test - y_pred_ols_smogn)

statistics = {
    "Paired t-test":
        mt.paired_ttest(
            ser_errors,
            smogn_errors,
        ),

    "Wilcoxon":
        mt.wilcoxon_test(
            ser_errors,
            smogn_errors,
        ),

    "Bootstrap":
        mt.bootstrap_confidence_interval(
            ser_errors,
            smogn_errors,
        ),

    "Cohen d":
        mt.cohens_d(
            ser_errors,
            smogn_errors,
        ),

    "Cliffs delta":
        mt.cliffs_delta(
            ser_errors,
            smogn_errors,
        ),
}

print("\n")
print("=" * 70)
print("STATISTICAL TESTS")
print("=" * 70)

for name, result in statistics.items():
    print(f"{name}:")
    print(result)
    print()
    
# =============================================================================
# 10. Save predictions
# =============================================================================

predictions = pd.DataFrame(
    {
        "Observed": y_test,
        "SER": y_pred_ser,
        "SMOGN_OLS": y_pred_ols_smogn,
        "OLS": y_pred_ols_plain,
    }
)

predictions.to_csv(
    RESULTS_DIR / "predictions.csv",
    index=False,
)
# =============================================================================
# 10. Save predictions
# =============================================================================

predictions = pd.DataFrame(
    {
        "Observed": y_test,
        "SER": y_pred_ser,
        "SMOGN_OLS": y_pred_ols_smogn,
        "OLS": y_pred_ols_plain,
    }
)

predictions.to_csv(
    RESULTS_DIR / "predictions.csv",
    index=False,
)