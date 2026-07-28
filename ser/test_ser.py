import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from metrics.metrics import regression_metrics
from ser import SERRegressor


# =====================================================
# Generate synthetic data
# =====================================================

np.random.seed(42)

X = pd.DataFrame({
    "x": np.random.uniform(-5, 5, 5000)
})

y = (
    X["x"] ** 2
    + np.random.normal(0, 2, len(X))
)

# =====================================================
# Train / Test split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
)

# =====================================================
# Train SER
# =====================================================

ser = SERRegressor(
    segmentation="A",
    verbose=True,
)

y_pred = ser.fit_predict(
    X_train,
    y_train,
    X_test,
)

metrics = regression_metrics(
    y_test,
    y_pred,
)

print("\nEvaluation")
print("-" * 50)

for name, value in metrics.items():
    print(f"{name:<15}: {value:.4f}")


    

