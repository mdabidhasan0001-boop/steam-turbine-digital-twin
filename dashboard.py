"""
Turbine Anomaly Detection Dashboard
====================================
A real-time Streamlit dashboard that loads a pre-trained Isolation Forest model
and simulates live turbine sensor readings with fault injection capability.

Run with: streamlit run dashboard.py
Requires: streamlit, scikit-learn, pandas, numpy, plotly, joblib
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import time
import os
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# PAGE CONFIG — must be the very first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Turbine AI Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLING — dark industrial theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* ---------- Google Font ---------- */
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');

  /* ---------- Base ---------- */
  html, body, [class*="css"] {
      font-family: 'Barlow', sans-serif;
  }
  .stApp {
      background: #0b0e14;
      color: #c8d6e5;
  }

  /* ---------- Hide Streamlit chrome ---------- */
  #MainMenu, footer, header { visibility: hidden; }

  /* ---------- Sidebar ---------- */
  [data-testid="stSidebar"] {
      background: #0f1520;
      border-right: 1px solid #1e2d45;
  }
  [data-testid="stSidebar"] * { color: #8fa8c8 !important; }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 { color: #4fc3f7 !important; }

  /* ---------- Metric cards ---------- */
  .metric-card {
      background: linear-gradient(135deg, #111827 0%, #0f1a28 100%);
      border: 1px solid #1e3a5f;
      border-radius: 10px;
      padding: 20px 24px 16px;
      position: relative;
      overflow: hidden;
      transition: border-color 0.4s;
  }
  .metric-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg, #0d6efd, #00b4d8);
  }
  .metric-card.fault::before {
      background: linear-gradient(90deg, #dc3545, #ff6b6b);
      animation: pulse-bar 1s ease-in-out infinite;
  }
  @keyframes pulse-bar {
      0%, 100% { opacity: 1; }
      50%       { opacity: 0.3; }
  }
  .metric-label {
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: #4a6fa5;
      margin-bottom: 8px;
  }
  .metric-value {
      font-family: 'Share Tech Mono', monospace;
      font-size: 2.2rem;
      color: #e2efff;
      line-height: 1;
  }
  .metric-unit {
      font-size: 13px;
      color: #4a6fa5;
      margin-left: 4px;
  }
  .metric-delta {
      font-size: 12px;
      margin-top: 6px;
      color: #52c41a;
  }
  .metric-delta.bad { color: #ff4d4f; }

  /* ---------- Status badge ---------- */
  .status-badge {
      display: inline-block;
      padding: 4px 14px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 2px;
      text-transform: uppercase;
      margin-top: 10px;
  }
  .status-normal {
      background: rgba(82, 196, 26, 0.15);
      border: 1px solid #52c41a;
      color: #52c41a;
  }
  .status-fault {
      background: rgba(255, 77, 79, 0.15);
      border: 1px solid #ff4d4f;
      color: #ff4d4f;
      animation: blink-badge 0.8s ease-in-out infinite;
  }
  @keyframes blink-badge {
      0%, 100% { opacity: 1; }
      50%       { opacity: 0.4; }
  }

  /* ---------- Section header ---------- */
  .section-header {
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 3px;
      text-transform: uppercase;
      color: #4a6fa5;
      margin: 28px 0 14px;
      padding-left: 4px;
      border-left: 3px solid #0d6efd;
      padding-left: 10px;
  }

  /* ---------- Header bar ---------- */
  .dashboard-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 18px 0 12px;
      border-bottom: 1px solid #1e2d45;
      margin-bottom: 24px;
  }
  .dashboard-title {
      font-size: 22px;
      font-weight: 700;
      color: #e2efff;
      letter-spacing: 0.5px;
  }
  .dashboard-subtitle {
      font-size: 12px;
      color: #4a6fa5;
      letter-spacing: 1px;
  }
  .live-dot {
      display: inline-block;
      width: 8px; height: 8px;
      border-radius: 50%;
      background: #52c41a;
      margin-right: 6px;
      animation: live-pulse 1.4s ease-in-out infinite;
  }
  @keyframes live-pulse {
      0%, 100% { box-shadow: 0 0 0 0 rgba(82,196,26,0.6); }
      50%       { box-shadow: 0 0 0 6px rgba(82,196,26,0); }
  }
  .live-label {
      font-size: 11px;
      color: #52c41a;
      font-weight: 600;
      letter-spacing: 2px;
  }

  /* ---------- Chart wrapper ---------- */
  .chart-wrapper {
      background: #0f1520;
      border: 1px solid #1e2d45;
      border-radius: 10px;
      padding: 8px;
      margin-bottom: 16px;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SENSOR THRESHOLDS (engineering limits)
# ─────────────────────────────────────────────
THRESHOLDS = {
    "rotor_rpm":    {"normal": (2800, 3200), "fault_mean": 3600, "fault_std": 120,  "unit": "RPM",  "label": "Rotor Speed"},
    "vibration_mm": {"normal": (0.5,  2.0),  "fault_mean": 5.5,  "fault_std": 0.8,  "unit": "mm/s", "label": "Vibration"},
    "exhaust_temp": {"normal": (420,  480),  "fault_mean": 560,  "fault_std": 15,   "unit": "°C",   "label": "Exhaust Temp"},
}
HISTORY_LEN = 120   # number of data-points kept in the rolling chart
REFRESH_MS  = 600   # milliseconds between updates


# ─────────────────────────────────────────────
# MODEL LOADER — cached so it runs only once
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """
    Load the pre-trained Isolation Forest from disk.
    Returns (model, True) on success, (None, False) if file missing.
    A simple stub model is created automatically when the real file
    is absent, so the dashboard is still fully demo-able.
    """
    if os.path.exists(path):
        try:
            bundle = joblib.load(path)
            return bundle["model"], bundle["scaler"], True
        except Exception as e:
            st.warning(f"Could not load model: {e}. Using demo stub.")

    # ── Demo stub: train a quick Isolation Forest on synthetic normal data
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    rng   = np.random.default_rng(42)
    X_fit = np.column_stack([
        rng.normal(3000, 80,  2000),   # rotor RPM
        rng.normal(1.2,  0.3, 2000),   # vibration mm/s
        rng.normal(450,  10,  2000),   # exhaust °C
    ])
    scaler = StandardScaler()
    scaler.fit(X_fit)
    stub = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    stub.fit(scaler.transform(X_fit))
    return stub, scaler, False


# ─────────────────────────────────────────────
# SESSION STATE — persists across reruns
# ─────────────────────────────────────────────
def init_state():
    """Initialise all session-state variables on first load."""
    defaults = {
        "step":       0,
        "history":    pd.DataFrame(columns=["ts", "rotor_rpm", "vibration_mm", "exhaust_temp", "anomaly"]),
        "last_score": 0.0,
        "is_fault":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
# SENSOR SIMULATION
# ─────────────────────────────────────────────
def generate_reading(step: int, inject_fault: bool) -> dict:
    """
    Produce one synthetic sensor reading.
    - Normal mode:  values drawn from realistic Gaussian distributions.
    - Fault mode:   after FAULT_ONSET steps the values drift toward
                    their fault distributions to mimic a developing fault.
    """
    FAULT_ONSET = 100  # steps after which fault values kick in

    rng = np.random.default_rng(seed=step)   # seeded so reruns are deterministic

    if inject_fault and step >= FAULT_ONSET:
        # Gradual drift: blend from normal → fault over 40 steps
        blend = min(1.0, (step - FAULT_ONSET) / 40.0)
        rpm   = rng.normal(3000 + blend * 600,  80  + blend * 120)
        vib   = rng.normal(1.2  + blend * 4.3,  0.3 + blend * 0.8)
        temp  = rng.normal(450  + blend * 110,  10  + blend * 15)
    else:
        rpm  = rng.normal(3000, 80)
        vib  = rng.normal(1.2,  0.3)
        temp = rng.normal(450,  10)

    return {
        "rotor_rpm":    round(float(np.clip(rpm,  2400, 4000)), 1),
        "vibration_mm": round(float(np.clip(vib,  0.0,  10.0)), 3),
        "exhaust_temp": round(float(np.clip(temp, 350,  650)),  1),
    }


def classify(model, scaler, reading: dict) -> tuple[bool, float]:
    """
    Run the Isolation Forest on the current reading.
    Returns (is_anomaly: bool, anomaly_score: float).
    Score > 0 = normal, score < 0 = anomalous (sklearn convention).
    """
    X = np.array([[reading["rotor_rpm"],
                   reading["vibration_mm"],
                   reading["exhaust_temp"]]])
    X = scaler.transform(X)         # apply same scaling used at training time
    label = model.predict(X)[0]        # +1 normal, -1 anomaly
    score = model.score_samples(X)[0]  # raw decision function value
    return (label == -1), score


# ─────────────────────────────────────────────
# CHART BUILDER — Plotly dark theme
# ─────────────────────────────────────────────
CHART_CFG = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font         =dict(family="Barlow, sans-serif", color="#8fa8c8", size=11),
    margin       =dict(l=48, r=16, t=36, b=36),
    height       =200,
    xaxis        =dict(showgrid=False, showline=False, zeroline=False,
                       tickfont=dict(size=10)),
    yaxis        =dict(gridcolor="#1e2d45", gridwidth=1,
                       showline=False, zeroline=False,
                       tickfont=dict(size=10)),
    hovermode    ="x unified",
    hoverlabel   =dict(bgcolor="#0f1520", font_color="#c8d6e5",
                       bordercolor="#1e3a5f"),
)

def make_chart(df: pd.DataFrame, col: str, label: str,
               unit: str, threshold_hi: float, line_color: str) -> go.Figure:
    """
    Build a single Plotly line chart for one sensor with:
      • A gradient-filled area under the line
      • A red dashed threshold line
      • Red markers on anomalous points
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_CFG, title=dict(text=f"{label}", x=0.01,
                          font=dict(size=12, color="#c8d6e5")))
        return fig

    normal_mask  = df["anomaly"] == False
    anomaly_mask = df["anomaly"] == True

    fig = go.Figure()

    # ── Filled area trace
    fig.add_trace(go.Scatter(
        x=df["ts"], y=df[col],
        mode="lines",
        name=label,
        line=dict(color=line_color, width=2),
        fill="tozeroy",
        fillcolor=line_color.replace("rgb", "rgba").replace(")", ", 0.08)"),
        hovertemplate=f"<b>{label}</b>: %{{y:.2f}} {unit}<extra></extra>",
    ))

    # ── Anomaly markers (red dots)
    if anomaly_mask.any():
        fig.add_trace(go.Scatter(
            x=df.loc[anomaly_mask, "ts"],
            y=df.loc[anomaly_mask, col],
            mode="markers",
            name="Anomaly",
            marker=dict(color="#ff4d4f", size=7, symbol="circle-open",
                        line=dict(width=2, color="#ff4d4f")),
            hovertemplate=f"<b>⚠ ANOMALY</b>: %{{y:.2f}} {unit}<extra></extra>",
        ))

    # ── Threshold line
    fig.add_hline(
        y=threshold_hi,
        line=dict(color="#ff4d4f", width=1.5, dash="dot"),
        annotation_text=f"  Limit {threshold_hi} {unit}",
        annotation_font=dict(size=10, color="#ff4d4f"),
        annotation_position="top left",
    )

    fig.update_layout(
        **CHART_CFG,
        title=dict(text=f"{label}  <span style='color:#4a6fa5;font-size:10px'>"
                        f"latest: {df[col].iloc[-1]:.2f} {unit}</span>",
                   x=0.01, font=dict(size=12, color="#c8d6e5")),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Turbine Monitor")
    st.markdown("---")

    inject_fault = st.checkbox(
        "💥 Inject Fault",
        value=False,
        help="Enables gradual sensor drift after 100 simulation steps "
             "to test the anomaly detection model.",
    )

    st.markdown("---")
    st.markdown("### 📊 Thresholds")
    st.caption("These are the engineering red-lines used for the chart annotations.")
    st.markdown(f"• Rotor Speed: **3200 RPM**")
    st.markdown(f"• Vibration:   **2.0 mm/s**")
    st.markdown(f"• Exhaust Temp: **480 °C**")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    refresh_rate = st.slider(
        "Refresh interval (ms)", min_value=200, max_value=2000,
        value=REFRESH_MS, step=100,
        help="How fast the simulation ticks forward."
    )
    history_len = st.slider(
        "Chart history (steps)", min_value=30, max_value=300,
        value=HISTORY_LEN, step=10,
    )

    st.markdown("---")
    model_path = st.text_input("Model path", value="turbine_model.pkl",
                                help="Path to your joblib-serialised Isolation Forest.")

    if st.button("🔄 Reset Simulation"):
        st.session_state.step    = 0
        st.session_state.history = pd.DataFrame(
            columns=["ts", "rotor_rpm", "vibration_mm", "exhaust_temp", "anomaly"])
        st.session_state.last_score = 0.0
        st.session_state.is_fault   = False
        st.rerun()


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model, scaler, model_loaded = load_model(model_path)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="dashboard-header">
  <div>
    <div class="dashboard-title">⚡ Turbine Anomaly Detection</div>
    <div class="dashboard-subtitle">
      {'MODEL LOADED · turbine_model.pkl' if model_loaded else 'DEMO MODEL · place turbine_model.pkl to use your own'}
      &nbsp;·&nbsp; ISOLATION FOREST
    </div>
  </div>
  <div>
    <span class="live-dot"></span>
    <span class="live-label">LIVE &nbsp;·&nbsp; STEP {st.session_state.step}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIMULATE ONE STEP
# ─────────────────────────────────────────────
reading   = generate_reading(st.session_state.step, inject_fault)
is_fault, score = classify(model, scaler, reading)

# Append to rolling history
new_row = {
    "ts":           datetime.now(),
    "rotor_rpm":    reading["rotor_rpm"],
    "vibration_mm": reading["vibration_mm"],
    "exhaust_temp": reading["exhaust_temp"],
    "anomaly":      is_fault,
}
st.session_state.history = pd.concat(
    [st.session_state.history, pd.DataFrame([new_row])],
    ignore_index=True
).tail(history_len)

st.session_state.step       += 1
st.session_state.last_score  = score
st.session_state.is_fault    = is_fault


# ─────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Live Sensor Readings</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

def card(col, label, value, unit, low, high, is_fault_card=False):
    """Render a single metric card."""
    in_range = low <= value <= high
    delta_class = "" if in_range else "bad"
    delta_icon  = "✓ Normal range" if in_range else "⚠ Out of range"
    fault_class = "fault" if is_fault_card else ""
    col.markdown(f"""
    <div class="metric-card {fault_class}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>
      <div class="metric-delta {delta_class}">{delta_icon}</div>
    </div>
    """, unsafe_allow_html=True)

with c1:
    card(c1, "Rotor Speed",  reading["rotor_rpm"],    "RPM",  2800, 3200)
with c2:
    card(c2, "Vibration",    reading["vibration_mm"], "mm/s", 0.5,  2.0)
with c3:
    card(c3, "Exhaust Temp", reading["exhaust_temp"], "°C",   420,  480)
with c4:
    # AI Status card — special rendering
    badge_cls   = "status-fault" if is_fault else "status-normal"
    badge_label = "⚠ FAULT DETECTED" if is_fault else "● NORMAL"
    card_cls    = "fault" if is_fault else ""
    score_disp  = f"{score:.4f}"
    c4.markdown(f"""
    <div class="metric-card {card_cls}">
      <div class="metric-label">AI Status</div>
      <div class="metric-value" style="font-size:1.5rem; margin-top:4px">
        {'FAULT' if is_fault else 'NORMAL'}
      </div>
      <div class="metric-delta {'bad' if is_fault else ''}">
        Score: {score_disp}
      </div>
      <span class="status-badge {badge_cls}">{badge_label}</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LIVE CHARTS
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Sensor Trends</div>',
            unsafe_allow_html=True)

df = st.session_state.history

chart_specs = [
    ("rotor_rpm",    "Rotor Speed",  "RPM",  3200, "rgb(13, 110, 253)"),
    ("vibration_mm", "Vibration",    "mm/s", 2.0,  "rgb(0, 180, 216)"),
    ("exhaust_temp", "Exhaust Temp", "°C",   480,  "rgb(255, 165, 2)"),
]

col_a, col_b, col_c = st.columns(3)

for col_widget, (field, label, unit, thresh, color) in zip(
        [col_a, col_b, col_c], chart_specs):
    fig = make_chart(df, field, label, unit, thresh, color)
    with col_widget:
        st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FAULT INJECTION STATUS BANNER
# ─────────────────────────────────────────────
if inject_fault:
    steps_until_fault = max(0, 100 - st.session_state.step)
    if steps_until_fault > 0:
        st.info(f"⏳ Fault injection armed — activates in **{steps_until_fault}** steps.")
    else:
        st.error("💥 **Fault injection ACTIVE** — sensor values drifting toward fault region.")


# ─────────────────────────────────────────────
# ANOMALY LOG (last 10 flagged events)
# ─────────────────────────────────────────────
anomalies = df[df["anomaly"] == True]
if not anomalies.empty:
    st.markdown('<div class="section-header">Recent Anomaly Events</div>',
                unsafe_allow_html=True)
    display_df = anomalies.tail(10)[["ts", "rotor_rpm", "vibration_mm", "exhaust_temp"]].copy()
    display_df["ts"] = display_df["ts"].dt.strftime("%H:%M:%S")
    display_df.columns = ["Timestamp", "Rotor RPM", "Vibration mm/s", "Exhaust °C"]
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# ─────────────────────────────────────────────
# AUTO-RERUN — keeps the dashboard ticking
# Must be the LAST statement in the script.
# ─────────────────────────────────────────────
time.sleep(refresh_rate / 1000)
st.rerun()
