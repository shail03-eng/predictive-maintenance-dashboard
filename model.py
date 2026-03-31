"""
Predictive Maintenance Model
============================
Trains a Random Forest Classifier to predict machine failures
using the AI4I 2020 Predictive Maintenance Dataset.

Author: Shail Vaghela
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os


# ── 1. Load Data ─────────────────────────────────────────────────────────────

def load_data(filepath="data/predictive_maintenance.csv"):
    """Load and return the dataset."""
    if not os.path.exists(filepath):
        print("Dataset not found. Downloading from UCI...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
        df = pd.read_csv(url)
        os.makedirs("data", exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
    else:
        df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ── 2. Explore Data ───────────────────────────────────────────────────────────

def explore_data(df):
    """Print key statistics about the dataset."""
    print("\n📊 Dataset Overview:")
    print(df.head())
    print("\n📋 Column Types:")
    print(df.dtypes)
    print("\n🔍 Missing Values:")
    print(df.isnull().sum())
    print("\n⚠️  Failure Distribution:")
    print(df["Machine failure"].value_counts())
    print(f"\n  Failure rate: {df['Machine failure'].mean()*100:.2f}%")


# ── 3. Preprocess Data ────────────────────────────────────────────────────────

def preprocess(df):
    """Clean and prepare features for model training."""

    # Encode product type (L, M, H → 0, 1, 2)
    le = LabelEncoder()
    df["Type_encoded"] = le.fit_transform(df["Type"])

    # Select features
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

    print(f"\n✅ Features selected: {features}")
    print(f"✅ Target: Machine failure (0 = No failure, 1 = Failure)")

    return X, y, features


# ── 4. Train Model ────────────────────────────────────────────────────────────

def train_model(X, y):
    """Split data and train a Random Forest Classifier."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"  # handles class imbalance
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n🎯 Model Performance:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"]))

    return model, X_test, y_test, y_pred


# ── 5. Visualize Results ──────────────────────────────────────────────────────

def plot_results(model, X_test, y_test, y_pred, features):
    """Generate and save visualizations."""

    os.makedirs("plots", exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Failure", "Failure"],
                yticklabels=["No Failure", "Failure"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png")
    plt.close()
    print("✅ Confusion matrix saved to plots/confusion_matrix.png")

    # Feature Importance
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": features, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(feat_df["Feature"], feat_df["Importance"], color="#2E75B6")
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png")
    plt.close()
    print("✅ Feature importance saved to plots/feature_importance.png")


# ── 6. Save Model ─────────────────────────────────────────────────────────────

def save_model(model, filepath="model.pkl"):
    """Save trained model to disk."""
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"\n💾 Model saved to {filepath}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  🔧 Predictive Maintenance Model Training")
    print("=" * 55)

    df = load_data()
    explore_data(df)
    X, y, features = preprocess(df)
    model, X_test, y_test, y_pred = train_model(X, y)
    plot_results(model, X_test, y_test, y_pred, features)
    save_model(model)

    print("\n✅ Training complete! Run 'streamlit run app.py' to launch the dashboard.")
