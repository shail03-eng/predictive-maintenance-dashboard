"""
Predictive Maintenance Dashboard
==================================
Interactive Streamlit dashboard to visualize machine health
and predict failure risk in real time.

Author: Shail Vaghela
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide"
)

# ── Load or Train Model ───────────────────────────────────────────────────────

@st.cache_resource
def load_model_and_data():
    """Load dataset and train model (cached for performance)."""
    
    # Load data
    if os.path.exists("data/predictive_maintenance.csv"):
        df = pd.read_csv("data/predictive_maintenance.csv")
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
        df = pd.read_csv(url)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/predictive_maintenance.csv", index=False)

    # Preprocess
    le = LabelEncoder()
    df["Type_encoded"] = le.fit_transform(df["Type"])

    features = [
        "Type_encoded",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    X = df[features]
    y = df["Machine failure"]

    # Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    return model, df, features, X_test, y_test


model, df, features, X_test, y_test = load_model_and_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🔧 Machine Parameters")
st.sidebar.markdown("Adjust the sliders to simulate machine conditions:")

machine_type = st.sidebar.selectbox(
    "Machine Type", options=["L (Light)", "M (Medium)", "H (Heavy)"]
)
type_map = {"L (Light)": 0, "M (Medium)": 1, "H (Heavy)": 2}
type_encoded = type_map[machine_type]

air_temp = st.sidebar.slider(
    "Air Temperature (K)", min_value=295, max_value=305, value=300
)
process_temp = st.sidebar.slider(
    "Process Temperature (K)", min_value=305, max_value=315, value=310
)
rot_speed = st.sidebar.slider(
    "Rotational Speed (RPM)", min_value=1168, max_value=2886, value=1500
)
torque = st.sidebar.slider(
    "Torque (Nm)", min_value=3, max_value=77, value=40
)
tool_wear = st.sidebar.slider(
    "Tool Wear (min)", min_value=0, max_value=253, value=100
)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🔧 Predictive Maintenance Dashboard")
st.markdown(
    "**Predict machine failures before they happen** — using Machine Learning on the AI4I 2020 dataset."
)
st.markdown("---")

# ── Prediction ────────────────────────────────────────────────────────────────

input_data = pd.DataFrame([[
    type_encoded, air_temp, process_temp, rot_speed, torque, tool_wear
]], columns=features)

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🌡️ Air Temperature", f"{air_temp} K")
    st.metric("⚙️ Rotational Speed", f"{rot_speed} RPM")

with col2:
    st.metric("🔥 Process Temperature", f"{process_temp} K")
    st.metric("🔩 Torque", f"{torque} Nm")

with col3:
    st.metric("🛠️ Tool Wear", f"{tool_wear} min")
    st.metric("🏭 Machine Type", machine_type)

st.markdown("---")

# Failure prediction result
st.subheader("⚡ Failure Prediction")

risk_col1, risk_col2 = st.columns(2)

with risk_col1:
    if prediction == 1:
        st.error(f"⚠️  **FAILURE RISK DETECTED**\n\nFailure probability: **{probability*100:.1f}%**")
    else:
        st.success(f"✅ **MACHINE OPERATING NORMALLY**\n\nFailure probability: **{probability*100:.1f}%**")

with risk_col2:
    # Risk gauge bar
    fig, ax = plt.subplots(figsize=(5, 1.2))
    color = "#e74c3c" if probability > 0.5 else "#2ecc71"
    ax.barh(["Risk Level"], [probability], color=color, height=0.5)
    ax.barh(["Risk Level"], [1 - probability], left=[probability],
            color="#ecf0f1", height=0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Failure Probability", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Dataset Insights ──────────────────────────────────────────────────────────

st.subheader("📊 Dataset Insights")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Failure distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    failure_counts = df["Machine failure"].value_counts()
    ax.bar(["No Failure", "Failure"], failure_counts.values,
           color=["#2E75B6", "#e74c3c"])
    ax.set_title("Failure Distribution in Dataset")
    ax.set_ylabel("Count")
    for i, v in enumerate(failure_counts.values):
        ax.text(i, v + 20, str(v), ha="center", fontweight="bold")
    st.pyplot(fig)
    plt.close()

with chart_col2:
    # Feature importance
    importances = model.feature_importances_
    feat_labels = [
        "Type", "Air Temp", "Process Temp",
        "Rot. Speed", "Torque", "Tool Wear"
    ]
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(
        [feat_labels[i] for i in sorted_idx],
        importances[sorted_idx],
        color="#2E75B6"
    )
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Raw Data Preview ──────────────────────────────────────────────────────────

with st.expander("🔍 View Raw Dataset"):
    st.dataframe(df.head(50))
    st.caption(f"Showing first 50 rows of {len(df)} total records.")

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "Built by **Shail Vaghela** | Mechanical Engineering & Management Graduate | "
    "Python · Scikit-learn · Streamlit"
)
