"""
=============================================================
  Insurance Bundle Recommendation — solution.py
  Target: Purchased_Coverage_Bundle (10-class classification)
  Best Model: XGBoost (F1 Macro ≈ 0.555+)
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

    Incorporates all cleaning steps from Cleaning_Data.py:
      - Missing value handling (Employer_ID, Broker_ID, Acquisition_Channel,
        Region_Code, Deductible_Tier, Child_Dependents)
      - Outlier fixing (winsorization, log transforms)
      - Feature engineering (Total_Dependents, cyclical date encoding)
      - Categorical encoding (ordinal, binary, one-hot, frequency)
    """
    df = df.copy()

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
# Merges logic from Cleaning_Data.py and Compare_Models.py
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    import time
    from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix, classification_report
    )
    from xgboost import XGBClassifier

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

    print(f"✅ Label mapping ({len(label_mapping)} classes)")

    # Separate features and target
    drop_cols = ["User_ID", target_col, "Target_Encoded",
                 "Policy_Cancelled_Post_Purchase"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df["Target_Encoded"]

    print(f"✅ Features: {X.shape[1]} | Target classes: {y.nunique()}")
    print("=" * 70)

    # ─────────────────────────────────────
    # C. TRAIN XGBOOST (BEST F1 MACRO ≈ 0.577 CV → ~0.555 on test)
    # ─────────────────────────────────────
    best_model = XGBClassifier(
        n_estimators=200,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )

    # Cross-validate to confirm F1 Macro
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        best_model, X, y, cv=cv,
        scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
        n_jobs=-1
    )
    print(f"\n📊 XGBoost 5-Fold CV Results:")
    print(f"   Accuracy     : {cv_results['test_accuracy'].mean():.4f}")
    print(f"   F1 Macro     : {cv_results['test_f1_macro'].mean():.4f}")
    print(f"   Precision Mac: {cv_results['test_precision_macro'].mean():.4f}")
    print(f"   Recall Macro : {cv_results['test_recall_macro'].mean():.4f}")

    best_f1 = cv_results["test_f1_macro"].mean()

    # ─────────────────────────────────────
    # D. TRAIN ON FULL DATA & SAVE
    # ─────────────────────────────────────
    print(f"\n💾 Training XGBoost on FULL dataset ...")
    best_model.fit(X, y)
    joblib.dump(best_model, "model.pkl")
    print(f"✅ model.pkl saved (XGBoost trained on {X.shape[0]:,} rows)")

    # ─────────────────────────────────────
    # E. GENERATE SUBMISSION ON TEST DATA
    # ─────────────────────────────────────
    print("\n📤 Generating submission from test.csv ...")
    raw_test = pd.read_csv("test.csv")
    test_df = preprocess(raw_test)

    model = load_model()
    submission = predict(test_df, model)
    submission.to_csv("submission.csv", index=False)
    print(f"✅ submission.csv saved ({submission.shape[0]:,} predictions)")

    # ─────────────────────────────────────
    # F. CALCULATE FINAL HACKATHON SCORE
    # ─────────────────────────────────────
    size_mb = os.path.getsize("model.pkl") / (1024 * 1024)

    raw_test_bench = pd.read_csv("test.csv")
    start_time = time.time()
    test_bench = preprocess(raw_test_bench)
    _ = predict(test_bench, model)
    latency_s = time.time() - start_time

    size_penalty = max(0.5, 1 - size_mb / 200)
    latency_penalty = max(0.5, 1 - latency_s / 10)
    final_score = best_f1 * size_penalty * latency_penalty

    print("\n" + "=" * 70)
    print("🏁 FINAL HACKATHON SCORE")
    print("=" * 70)
    print(f"  Macro F1       : {best_f1:.4f}")
    print(f"  Model Size     : {size_mb:.2f} MB")
    print(f"  Latency        : {latency_s:.2f} s")
    print(f"  Size Penalty   : {size_penalty:.4f}")
    print(f"  Latency Penalty: {latency_penalty:.4f}")
    print(f"  🎯 Final Score : {final_score:.4f}")
    print("=" * 70)