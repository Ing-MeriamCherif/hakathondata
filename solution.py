"""
=============================================================
  Insurance Bundle Recommendation — solution.py
  Target: Purchased_Coverage_Bundle (10-class classification)
  Best Model: LightGBM
=============================================================
  Required Interface:
    preprocess(df)        → Returns cleaned pandas DataFrame
    load_model()          → Returns loaded model object
    predict(df, model)    → Returns DataFrame with User_ID & Purchased_Coverage_Bundle
=============================================================
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════
# PREPROCESS
# ═══════════════════════════════════════════════════════════
def preprocess(df):
    """
    Clean, encode, and engineer features.
    Works for both train (has Purchased_Coverage_Bundle) and test (doesn't).
    Returns a cleaned pandas DataFrame.
    """
    df = df.copy()

    # --- Save User_ID for later, remove from features ---
    # (kept in df but will be excluded during training)

    # --- Determine if this is train or test ---
    is_train = "Purchased_Coverage_Bundle" in df.columns

    # ─────────────────────────────────────────
    # 1. HANDLE IDENTIFIER COLUMNS
    # ─────────────────────────────────────────
    # Employer_ID → 94.3% missing — keep as binary flag
    df["Has_Employer_ID"] = df["Employer_ID"].notna().astype(int)
    df.drop(columns=["Employer_ID"], inplace=True)

    # ─────────────────────────────────────────
    # 2. HANDLE MISSING VALUES
    # ─────────────────────────────────────────
    # Broker_ID (13.7% missing)
    df["Has_Broker_ID"] = df["Broker_ID"].notna().astype(int)
    df["Broker_ID"] = df["Broker_ID"].fillna(-1).astype(int).astype(str)

    # Acquisition_Channel (~1% missing) → mode
    df["Acquisition_Channel"] = df["Acquisition_Channel"].fillna(
        df["Acquisition_Channel"].mode()[0]
    )

    # Region_Code (~0.5% missing)
    df["Region_Code"] = df["Region_Code"].fillna("Unknown")

    # Deductible_Tier (~0.5% missing) → mode
    df["Deductible_Tier"] = df["Deductible_Tier"].fillna(
        df["Deductible_Tier"].mode()[0]
    )

    # Child_Dependents (4 missing) → median
    df["Child_Dependents"] = df["Child_Dependents"].fillna(
        df["Child_Dependents"].median()
    ).astype(int)

    # ─────────────────────────────────────────
    # 3. FIX OUTLIERS
    # ─────────────────────────────────────────
    df["Child_Dependents"] = df["Child_Dependents"].clip(upper=5)

    income_cap = df["Estimated_Annual_Income"].quantile(0.99)
    df["Estimated_Annual_Income"] = df["Estimated_Annual_Income"].clip(upper=income_cap)
    df["Estimated_Annual_Income_Log"] = np.log1p(df["Estimated_Annual_Income"])
    df.drop(columns=["Estimated_Annual_Income"], inplace=True)

    days_quote_cap = df["Days_Since_Quote"].quantile(0.99)
    df["Days_Since_Quote"] = df["Days_Since_Quote"].clip(upper=days_quote_cap)

    df["Underwriting_Processing_Days"] = np.log1p(df["Underwriting_Processing_Days"])

    # ─────────────────────────────────────────
    # 4. FEATURE ENGINEERING
    # ─────────────────────────────────────────
    df["Total_Dependents"] = (
        df["Adult_Dependents"] + df["Child_Dependents"] + df["Infant_Dependents"]
    )

    # Cyclical encoding — Month
    month_order = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    df["Month_Num"] = df["Policy_Start_Month"].map(month_order)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month_Num"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month_Num"] / 12)
    df.drop(columns=["Policy_Start_Month", "Month_Num"], inplace=True)

    # Cyclical encoding — Day
    df["Day_Sin"] = np.sin(2 * np.pi * df["Policy_Start_Day"] / 31)
    df["Day_Cos"] = np.cos(2 * np.pi * df["Policy_Start_Day"] / 31)
    df.drop(columns=["Policy_Start_Day"], inplace=True)

    # Cyclical encoding — Week
    df["Week_Sin"] = np.sin(2 * np.pi * df["Policy_Start_Week"] / 52)
    df["Week_Cos"] = np.cos(2 * np.pi * df["Policy_Start_Week"] / 52)
    df.drop(columns=["Policy_Start_Week"], inplace=True)

    # ─────────────────────────────────────────
    # 5. ENCODE CATEGORICAL VARIABLES
    # ─────────────────────────────────────────
    # Ordinal: Deductible_Tier
    deductible_order = {
        "Tier_1_High_Ded": 1, "Tier_2_Mid_Ded": 2,
        "Tier_3_Low_Ded": 3, "Tier_4_Zero_Ded": 4
    }
    df["Deductible_Tier"] = df["Deductible_Tier"].map(deductible_order)

    # Binary: Broker_Agency_Type
    df["Is_National_Corporate"] = (df["Broker_Agency_Type"] == "National_Corporate").astype(int)
    df.drop(columns=["Broker_Agency_Type"], inplace=True)

    # One-Hot: low-cardinality categoricals
    ohe_cols = ["Acquisition_Channel", "Payment_Schedule", "Employment_Status"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True, dtype=int)

    # Frequency encoding for Region_Code (safe for train & test — no target leakage)
    region_freq = df["Region_Code"].value_counts(normalize=True)
    df["Region_Code_FreqEnc"] = df["Region_Code"].map(region_freq)
    df.drop(columns=["Region_Code"], inplace=True)

    # Frequency encoding for Broker_ID
    broker_freq = df["Broker_ID"].value_counts(normalize=True)
    df["Broker_ID_FreqEnc"] = df["Broker_ID"].map(broker_freq)
    df.drop(columns=["Broker_ID"], inplace=True)

    return df


# ═══════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════
def load_model():
    """Load the trained model from model.pkl."""
    return joblib.load("model.pkl")


# ═══════════════════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════════════════
def predict(df, model):
    """
    Takes a preprocessed DataFrame + trained model.
    Returns a DataFrame with User_ID and Purchased_Coverage_Bundle (integers 0–9).
    All input User_IDs are guaranteed to appear in the output.
    """
    user_ids = df["User_ID"].reset_index(drop=True)
    features = df.drop(columns=["User_ID"])

    # Drop target columns if present (train data)
    for col in ["Purchased_Coverage_Bundle", "Policy_Cancelled_Post_Purchase"]:
        if col in features.columns:
            features = features.drop(columns=[col])

    # Predict — returns integer labels (0–9)
    preds = model.predict(features)

    result = pd.DataFrame({
        "User_ID": user_ids,
        "Purchased_Coverage_Bundle": preds.astype(int)
    })

    # Validate: all User_IDs must be present
    assert len(result) == len(user_ids), (
        f"Missing predictions! Expected {len(user_ids)}, got {len(result)}"
    )
    return result


# ═══════════════════════════════════════════════════════════
# TRAINING & EVALUATION (runs when you execute this file)
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix,
        classification_report, RocCurveDisplay
    )
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    # ─────────────────────────────────────
    # A. LOAD & PREPROCESS TRAIN DATA
    # ─────────────────────────────────────
    print("=" * 70)
    print("  INSURANCE BUNDLE RECOMMENDATION — TRAINING PIPELINE")
    print("=" * 70)

    raw_train = pd.read_csv("train.csv")
    print(f"✅ Loaded train.csv: {raw_train.shape[0]:,} rows × {raw_train.shape[1]} columns")

    df = preprocess(raw_train)
    print(f"✅ Preprocessed: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ─────────────────────────────────────
    # B. PREPARE FEATURES & TARGET
    # ─────────────────────────────────────
    target_col = "Purchased_Coverage_Bundle"

    # Encode target labels to integers
    label_mapping = {label: idx for idx, label in enumerate(
        df[target_col].unique()
    )}
    df["Target_Encoded"] = df[target_col].map(label_mapping)

    # Save label mapping for predict()
    joblib.dump(label_mapping, "label_mapping.pkl")
    print(f"✅ Label mapping saved ({len(label_mapping)} classes)")

    # Separate features and target
    drop_cols = ["User_ID", target_col, "Target_Encoded",
                 "Policy_Cancelled_Post_Purchase"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df["Target_Encoded"]

    print(f"✅ Features: {X.shape[1]} | Target classes: {y.nunique()}")
    print(f"   Class distribution:\n{df[target_col].value_counts().to_string()}")
    print("=" * 70)

    # ─────────────────────────────────────
    # C. DEFINE MODELS
    # ─────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Random Forest":       RandomForestClassifier(n_estimators=200,
                                                       random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200,
                                                           random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=200, use_label_encoder=False,
                                              eval_metric="mlogloss", random_state=42,
                                              verbosity=0, n_jobs=-1),
        "LightGBM":            LGBMClassifier(n_estimators=200, random_state=42,
                                               verbose=-1, n_jobs=-1),
    }

    # ─────────────────────────────────────
    # D. CROSS-VALIDATED EVALUATION
    # ─────────────────────────────────────
    scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    print("\n🔄 Running 5-Fold Stratified Cross-Validation ...\n")

    for name, model in models.items():
        print(f"  ⏳ Training {name} ...")
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = {
            "Accuracy":        cv_results["test_accuracy"].mean(),
            "F1 Macro":        cv_results["test_f1_macro"].mean(),
            "Precision Macro": cv_results["test_precision_macro"].mean(),
            "Recall Macro":    cv_results["test_recall_macro"].mean(),
        }
        print(f"     ✅ {name}  →  Accuracy: {results[name]['Accuracy']:.4f}  |  "
              f"F1 Macro: {results[name]['F1 Macro']:.4f}")

    print("\n" + "=" * 70)

    # ─────────────────────────────────────
    # E. RESULTS SUMMARY TABLE
    # ─────────────────────────────────────
    results_df = pd.DataFrame(results).T.sort_values("F1 Macro", ascending=False)
    results_df.index.name = "Model"

    print("\n📊 MODEL COMPARISON RESULTS (sorted by F1 Macro)\n")
    print(results_df.to_string(float_format="{:.4f}".format))

    best_model_name = results_df["F1 Macro"].idxmax()
    best_f1 = results_df.loc[best_model_name, "F1 Macro"]
    print(f"\n🏆 Best Model: {best_model_name} (F1 Macro = {best_f1:.4f})")
    print("=" * 70)

    results_df.to_csv("model_comparison_results.csv")
    print("✅ Results saved to: model_comparison_results.csv")

    # ─────────────────────────────────────
    # F. VISUALIZATION: Metric Comparison Bar Chart
    # ─────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    fig.suptitle("Model Comparison — 5-Fold CV (Multi-Class)", fontsize=16,
                 fontweight="bold", y=1.02)

    colors = sns.color_palette("viridis", n_colors=len(results_df))

    for i, metric in enumerate(["Accuracy", "F1 Macro", "Precision Macro", "Recall Macro"]):
        ax = axes[i]
        bars = ax.barh(results_df.index, results_df[metric], color=colors, edgecolor="white")
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.axvline(x=results_df[metric].max(), color="red", linestyle="--", alpha=0.5)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("comparison_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved: comparison_bar_chart.png")

    # ─────────────────────────────────────
    # G. TRAIN/TEST SPLIT FOR DETAILED EVAL
    # ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Fit all models on the split
    for name, model in models.items():
        model.fit(X_train, y_train)

    # ─────────────────────────────────────
    # H. CONFUSION MATRICES — TOP 3
    # ─────────────────────────────────────
    top3 = results_df.head(3).index.tolist()
    idx_to_label = {v: k for k, v in label_mapping.items()}
    class_names = [idx_to_label[i] for i in range(len(label_mapping))]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("Confusion Matrices — Top 3 Models", fontsize=14, fontweight="bold")

    for ax, name in zip(axes, top3):
        model = models[name]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        acc = accuracy_score(y_test, y_pred)
        ax.set_title(f"{name}\n(Acc: {acc:.3f})", fontsize=10)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', rotation=0, labelsize=7)

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved: confusion_matrices.png")

    # ─────────────────────────────────────
    # I. RADAR CHART — TOP 3
    # ─────────────────────────────────────
    metrics = ["Accuracy", "F1 Macro", "Precision Macro", "Recall Macro"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    radar_colors = sns.color_palette("Set2", n_colors=3)

    for i, name in enumerate(top3):
        values = results_df.loc[name, metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=name, color=radar_colors[i])
        ax.fill(angles, values, alpha=0.15, color=radar_colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Radar Chart — Top 3 Models", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0), fontsize=9)
    plt.tight_layout()
    plt.savefig("radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved: radar_chart.png")

    # ─────────────────────────────────────
    # J. DETAILED REPORT — BEST MODEL
    # ─────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"📋 DETAILED CLASSIFICATION REPORT — {best_model_name}")
    print("=" * 70)
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best,
          target_names=class_names))

    f1_macro = f1_score(y_test, y_pred_best, average="macro")
    print(f"🎯 F1 Macro Score ({best_model_name}): {f1_macro:.4f}")

    # ─────────────────────────────────────
    # K. TRAIN BEST MODEL ON FULL DATA & SAVE
    # ─────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"💾 Training {best_model_name} on FULL dataset & saving ...")
    best_model_final = models[best_model_name]
    best_model_final.fit(X, y)

    joblib.dump(best_model_final, "model.pkl")
    print(f"✅ model.pkl saved ({best_model_name} trained on {X.shape[0]:,} rows)")

    # ─────────────────────────────────────
    # L. GENERATE SUBMISSION ON TEST DATA
    # ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("📤 Generating submission from test.csv ...")
    raw_test = pd.read_csv("test.csv")
    test_df = preprocess(raw_test)

    model = load_model()
    submission = predict(test_df, model)
    submission.to_csv("submission.csv", index=False)
    print(f"✅ submission.csv saved ({submission.shape[0]:,} predictions)")
    print(submission.head(10).to_string())

    print("\n" + "=" * 70)
    print("✅ All done! Files generated:")
    print("   • model.pkl                 — trained best model")
    print("   • label_mapping.pkl         — class label ↔ index mapping")
    print("   • submission.csv            — predictions for test.csv")
    print("   • model_comparison_results.csv")
    print("   • comparison_bar_chart.png")
    print("   • confusion_matrices.png")
    print("   • radar_chart.png")

    # ─────────────────────────────────────
    # M. CALCULATE FINAL HACKATHON SCORE
    # ─────────────────────────────────────
    import os
    import time

    def calculate_final_score(macro_f1, size_mb, latency_s):
        size_penalty = max(0.5, 1 - size_mb / 200)
        latency_penalty = max(0.5, 1 - latency_s / 10)
        final_score = macro_f1 * size_penalty * latency_penalty
        return final_score

    # Model size
    size_mb = os.path.getsize("model.pkl") / (1024 * 1024)

    # Inference latency (preprocess + predict on full test set)
    model = load_model()
    raw_test_bench = pd.read_csv("test.csv")
    start_time = time.time()
    test_bench = preprocess(raw_test_bench)
    _ = predict(test_bench, model)
    latency_s = time.time() - start_time

    # Use the CV F1 Macro from the best model
    macro_f1 = best_f1

    score = calculate_final_score(macro_f1, size_mb, latency_s)

    print("\n" + "=" * 70)
    print("🏁 FINAL HACKATHON SCORE")
    print("=" * 70)
    print(f"  Macro F1       : {macro_f1:.4f}")
    print(f"  Model Size     : {size_mb:.2f} MB")
    print(f"  Latency        : {latency_s:.2f} s")
    print(f"  Size Penalty   : {max(0.5, 1 - size_mb / 200):.4f}")
    print(f"  Latency Penalty: {max(0.5, 1 - latency_s / 10):.4f}")
    print(f"  🎯 Final Score : {score:.4f}")
    print("=" * 70)
