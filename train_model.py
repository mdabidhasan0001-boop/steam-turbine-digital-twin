"""
Turbine Fault Detection using Isolation Forest
===============================================
Isolation Forest is an unsupervised anomaly detection algorithm. Instead of
profiling normal behaviour, it exploits the fact that anomalies are *rare* and
*different*: it randomly partitions the feature space using decision trees and
measures how quickly each point gets "isolated". Points that are isolated in
fewer splits are flagged as anomalies, because unusual values need fewer cuts
to separate them from the rest of the data.

Why train only on normal data?
-------------------------------
This is the standard "one-class classification" approach. We want the model to
learn what *healthy* turbine operation looks like. If we trained on all data
(including faults), the model would learn a mixed distribution and would be
less sensitive to genuine deviations from normal. By showing it only normal
examples, the isolation boundary is tightly fitted around healthy behaviour —
anything outside that boundary is suspicious.
"""

import pickle
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("turbine_data.csv")

FEATURES = ["rotor_speed_rpm", "vibration_mm_s", "exhaust_temp_c"]
TARGET   = "fault"

X = df[FEATURES].values
y_true = df[TARGET].values          # ground-truth labels (0 = normal, 1 = fault)

# ── 2. Scale features ─────────────────────────────────────────────────────────
# Isolation Forest is not distance-based, but scaling keeps things consistent
# and makes the contamination estimate more reliable.
scaler  = StandardScaler()
X_normal = X[y_true == 0]           # ← only normal samples used for fitting
scaler.fit(X_normal)                # fit scaler on normal data only
X_scaled  = scaler.transform(X)     # transform the full dataset for prediction

# ── 3. Train Isolation Forest on NORMAL data only ────────────────────────────
# contamination="auto" tells the algorithm to use the theoretical 0.5 threshold
# from the original paper rather than assuming a fixed fault rate.
print("Training Isolation Forest on normal samples only...")
iso_forest = IsolationForest(
    n_estimators=200,       # number of isolation trees — more = more stable
    max_samples="auto",     # sub-sample size per tree (default: min(256, n))
    contamination="auto",   # decision threshold (no prior assumption on fault %)
    random_state=42,
    n_jobs=-1               # use all CPU cores
)
iso_forest.fit(X_scaled[y_true == 0])   # <-- training on normal rows only

# ── 4. Predict anomalies on the FULL dataset ─────────────────────────────────
# IsolationForest.predict() returns:
#   +1  → inlier  (normal)
#   -1  → outlier (anomaly / fault)
# We remap to match our labels: 0 = normal, 1 = fault
print("Predicting on full dataset...")
raw_preds = iso_forest.predict(X_scaled)
y_pred = (raw_preds == -1).astype(int)  # -1 → 1 (fault), +1 → 0 (normal)

# Anomaly scores (lower = more anomalous)
df["anomaly_score"] = iso_forest.decision_function(X_scaled)
df["predicted_fault"] = y_pred

# ── 5. Classification report ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Classification Report")
print("=" * 55)
print(classification_report(
    y_true,
    y_pred,
    target_names=["Normal (0)", "Fault (1)"]
))

# Quick summary counts
n_total  = len(df)
n_faults = y_pred.sum()
print(f"Total samples   : {n_total}")
print(f"Predicted faults: {n_faults}  ({100 * n_faults / n_total:.1f}%)")
print("=" * 55 + "\n")

# ── 6. Save model + scaler ────────────────────────────────────────────────────
# We bundle both objects so that inference code always applies the same scaling.
model_bundle = {
    "model":    iso_forest,
    "scaler":   scaler,
    "features": FEATURES,
}

MODEL_PATH = "turbine_model.pkl"
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_bundle, f)

print(f"Model saved to '{MODEL_PATH}'")
print("Bundle contains: model (IsolationForest), scaler (StandardScaler), feature list.")

# ── Example: how to reload and use the saved model ───────────────────────────
# with open("turbine_model.pkl", "rb") as f:
#     bundle = pickle.load(f)
# new_X = bundle["scaler"].transform(new_data[bundle["features"]])
# predictions = bundle["model"].predict(new_X)   # +1 normal / -1 fault
