# =============================================================================
# generate_data.py
# Steam Turbine Digital Twin — Synthetic SCADA Data Generator
# Simulates a 210 MW steam turbine with 3 sensors over time.
# Produces 1000 normal operation points + 50 injected fault points.
# Output: turbine_data.csv  (columns: timestamp, rotor_speed_rpm,
#         vibration_mm_s, exhaust_temp_c, fault_label)
# =============================================================================

import numpy as np                  # Numerical operations and random distributions
import pandas as pd                 # DataFrame construction and CSV export
from datetime import datetime, timedelta  # Realistic timestamp generation

# -----------------------------------------------------------------------------
# 1. REPRODUCIBILITY — fix the random seed so every run gives the same data
# -----------------------------------------------------------------------------
np.random.seed(42)                  # Any integer works; 42 is conventional

# -----------------------------------------------------------------------------
# 2. CONFIGURATION — all "magic numbers" live here for easy tuning
# -----------------------------------------------------------------------------

# --- Dataset size ---
N_NORMAL = 1000                     # Number of normal operating data points
N_FAULTS = 50                       # Number of fault data points to inject

# --- Sensor: Rotor Speed (RPM) ---
# A 2-pole turbine on a 50 Hz grid synchronises at exactly 3000 RPM.
# Real speed fluctuates slightly around the setpoint (~±5 RPM, 1-sigma).
SPEED_MEAN   = 3000.0               # Nominal synchronous speed (RPM)
SPEED_STD    =   5.0                # Normal operational noise (RPM, 1-sigma)
SPEED_FAULT_LOW  = 2800.0           # Under-speed fault threshold (RPM)
SPEED_FAULT_HIGH = 3200.0           # Over-speed fault threshold (RPM)

# --- Sensor: Vibration (mm/s RMS) ---
# ISO 10816-2 puts healthy large turbines below 4.5 mm/s.
# Normal baseline ≈ 2.5 mm/s with small Gaussian noise.
VIB_MEAN   = 2.5                    # Normal vibration level (mm/s)
VIB_STD    = 0.3                    # Normal operational noise (mm/s, 1-sigma)
VIB_FAULT_LOW  =  0.1               # Near-zero vib → sensor dropout fault
VIB_FAULT_HIGH = 15.0               # High vibration → imbalance / bearing fault

# --- Sensor: Exhaust Temperature (°C) ---
# LP exhaust is typically 40–100 °C, but here we model a back-pressure turbine
# exhausting at ~450 °C (process-heat application).
TEMP_MEAN  = 450.0                  # Normal exhaust temperature (°C)
TEMP_STD   =  10.0                  # Normal operational noise (°C, 1-sigma)
TEMP_FAULT_LOW  = 300.0             # Low temp → steam starvation / valve fault
TEMP_FAULT_HIGH = 600.0             # High temp → overload / cooling failure

# --- Timestamps ---
START_TIME  = datetime(2024, 1, 1, 0, 0, 0)   # Simulation start date-time
SAMPLE_INTERVAL = timedelta(seconds=10)        # One reading every 10 seconds

# -----------------------------------------------------------------------------
# 3. GENERATE NORMAL OPERATING DATA
# -----------------------------------------------------------------------------

# numpy.random.normal(mean, std, size) draws from a Gaussian distribution.
# Each sensor is sampled independently — no cross-correlation for simplicity.

rotor_speed_normal = np.random.normal(SPEED_MEAN, SPEED_STD,  N_NORMAL)
vibration_normal   = np.random.normal(VIB_MEAN,   VIB_STD,    N_NORMAL)
exhaust_temp_normal= np.random.normal(TEMP_MEAN,  TEMP_STD,   N_NORMAL)

# Clamp physically impossible negatives (vibration & temperature must be > 0)
vibration_normal    = np.clip(vibration_normal,    0.0, None)
exhaust_temp_normal = np.clip(exhaust_temp_normal, 0.0, None)

# Labels for normal data: 0 = healthy / no fault
labels_normal = np.zeros(N_NORMAL, dtype=int)

# -----------------------------------------------------------------------------
# 4. GENERATE FAULT DATA
# -----------------------------------------------------------------------------
# For each fault point we randomly choose ONE of three fault modes:
#   Mode A — Rotor speed anomaly  (over-speed or under-speed)
#   Mode B — Vibration anomaly    (high vibration or sensor dropout)
#   Mode C — Temperature anomaly  (overheating or steam starvation)
# The two unaffected sensors still read normally so the fault is realistic.

# Start from normal-looking background values for all three sensors
rotor_speed_fault  = np.random.normal(SPEED_MEAN, SPEED_STD, N_FAULTS)
vibration_fault    = np.random.normal(VIB_MEAN,   VIB_STD,   N_FAULTS)
exhaust_temp_fault = np.random.normal(TEMP_MEAN,  TEMP_STD,  N_FAULTS)

# Randomly assign each fault point to one of the three fault modes
fault_modes = np.random.choice(['speed', 'vibration', 'temperature'], size=N_FAULTS)

for i, mode in enumerate(fault_modes):

    if mode == 'speed':
        # Randomly pick either over-speed or under-speed condition
        if np.random.rand() > 0.5:
            # Over-speed: uniform random between high-fault boundary and +10%
            rotor_speed_fault[i] = np.random.uniform(
                SPEED_FAULT_HIGH, SPEED_FAULT_HIGH * 1.1
            )
        else:
            # Under-speed: uniform random between -10% and low-fault boundary
            rotor_speed_fault[i] = np.random.uniform(
                SPEED_FAULT_LOW * 0.9, SPEED_FAULT_LOW
            )

    elif mode == 'vibration':
        if np.random.rand() > 0.3:           # 70% chance → high-vibration event
            # Imbalance / bearing fault: severe spike above fault threshold
            vibration_fault[i] = np.random.uniform(
                VIB_FAULT_HIGH, VIB_FAULT_HIGH * 1.5
            )
        else:                                # 30% chance → sensor dropout
            vibration_fault[i] = np.random.uniform(
                VIB_FAULT_LOW, 0.5           # Near-zero reading
            )

    elif mode == 'temperature':
        if np.random.rand() > 0.4:           # 60% chance → overheating
            exhaust_temp_fault[i] = np.random.uniform(
                TEMP_FAULT_HIGH, TEMP_FAULT_HIGH * 1.15
            )
        else:                                # 40% chance → steam starvation
            exhaust_temp_fault[i] = np.random.uniform(
                TEMP_FAULT_LOW * 0.85, TEMP_FAULT_LOW
            )

# Labels for fault data: 1 = fault detected
labels_fault = np.ones(N_FAULTS, dtype=int)

# -----------------------------------------------------------------------------
# 5. COMBINE NORMAL + FAULT DATA
# -----------------------------------------------------------------------------

# Stack normal rows on top of fault rows using numpy concatenate
rotor_speed_all  = np.concatenate([rotor_speed_normal,  rotor_speed_fault])
vibration_all    = np.concatenate([vibration_normal,    vibration_fault])
exhaust_temp_all = np.concatenate([exhaust_temp_normal, exhaust_temp_fault])
labels_all       = np.concatenate([labels_normal,       labels_fault])

# Total number of rows in the final dataset
N_TOTAL = N_NORMAL + N_FAULTS       # 1050

# -----------------------------------------------------------------------------
# 6. BUILD TIMESTAMPS
# -----------------------------------------------------------------------------
# Create one timestamp per sample, spaced SAMPLE_INTERVAL apart.
# List comprehension: timestamp[i] = START_TIME + i × 10 seconds
timestamps = [START_TIME + i * SAMPLE_INTERVAL for i in range(N_TOTAL)]

# -----------------------------------------------------------------------------
# 7. ROUND SENSOR VALUES TO REALISTIC PRECISION
# -----------------------------------------------------------------------------
# Real SCADA historian values are not stored with 15 decimal places.
rotor_speed_all  = np.round(rotor_speed_all,  2)   # RPM     → 2 d.p.
vibration_all    = np.round(vibration_all,    3)   # mm/s    → 3 d.p.
exhaust_temp_all = np.round(exhaust_temp_all, 2)   # °C      → 2 d.p.

# -----------------------------------------------------------------------------
# 8. ASSEMBLE PANDAS DATAFRAME
# -----------------------------------------------------------------------------
df = pd.DataFrame({
    'timestamp'          : timestamps,        # ISO-8601 datetime string
    'rotor_speed_rpm'    : rotor_speed_all,   # Rotor speed in RPM
    'vibration_mm_s'     : vibration_all,     # Vibration in mm/s RMS
    'exhaust_temp_c'     : exhaust_temp_all,  # Exhaust temperature in °C
    'fault_label'        : labels_all         # 0 = normal, 1 = fault
})

# Shuffle the rows so fault points aren't all bunched at the end.
# reset_index(drop=True) gives a clean 0-based integer index after shuffle.
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# -----------------------------------------------------------------------------
# 9. SAVE TO CSV
# -----------------------------------------------------------------------------
OUTPUT_FILE = 'turbine_data.csv'
df.to_csv(OUTPUT_FILE, index=False)   # index=False → no extra row-number column

# -----------------------------------------------------------------------------
# 10. SUMMARY REPORT — printed to console after generation
# -----------------------------------------------------------------------------
print("=" * 60)
print("  Steam Turbine Synthetic SCADA Data — Generation Complete")
print("=" * 60)
print(f"  Output file    : {OUTPUT_FILE}")
print(f"  Total rows     : {len(df)}")
print(f"  Normal points  : {(df['fault_label'] == 0).sum()}")
print(f"  Fault  points  : {(df['fault_label'] == 1).sum()}")
print(f"  Date range     : {df['timestamp'].iloc[0]}  →  {df['timestamp'].iloc[-1]}")
print("-" * 60)
print("\nSensor Statistics (all rows):")
print(df[['rotor_speed_rpm', 'vibration_mm_s', 'exhaust_temp_c']].describe().round(3))
print("\nSensor Statistics (normal rows only):")
normal_df = df[df['fault_label'] == 0]
print(normal_df[['rotor_speed_rpm', 'vibration_mm_s', 'exhaust_temp_c']].describe().round(3))
print("\nSensor Statistics (fault rows only):")
fault_df = df[df['fault_label'] == 1]
print(fault_df[['rotor_speed_rpm', 'vibration_mm_s', 'exhaust_temp_c']].describe().round(3))
print("=" * 60)
print("  First 5 rows of the CSV:")
print("=" * 60)
print(df.head().to_string(index=False))
print("=" * 60)
