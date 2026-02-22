"""
=============================================================
  Insurance Bundle Recommendation — solution.py (FIXED)
  Target: Purchased_Coverage_Bundle (10-class classification)
  Model: XGBoost
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
import os
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════
# FIXED EXPECTED COLUMNS — hardcoded from training to guarantee
# train/test alignment. No more pd.get_dummies mismatch.
# ═══════════════════════════════════════════════════════════

# These will be set during training and saved inside model.pkl bundle
# At predict time they are loaded from the saved artifact.

MONTH_ORDER = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

DEDUCTIBLE_ORDER = {
    "Tier_1_High_Ded": 1, "Tier_2_Mid_Ded": 2,
    "Tier_3_Low_Ded": 3, "Tier_4_Zero_Ded": 4
}


# ═══════════════════════════════════════════════════════════
# PREPROCESS
# ═══════════════════════════════════════════════════════════
def preprocess(df):
    """
    Clean, encode, and engineer features.
    Works for both train (has Purchased_Coverage_Bundle) and test (doesn't).
    Returns a cleaned pandas DataFrame — keeps User_ID.
    """
    df = df.copy()

    # ─────────────────────────────────────────
    # 1. HANDLE IDENTIFIER COLUMNS
    # ─────────────────────────────────────────
    df["Has_Employer_ID"] = df["Employer_ID"].notna().astype(int)
    df.drop(columns=["Employer_ID"], inplace=True)

    # ─────────────────────────────────────────
    # 2. HANDLE MISSING VALUES
    # ─────────────────────────────────────────
    df["Has_Broker_ID"] = df["Broker_ID"].notna().astype(int)
    df["Broker_ID"] = df["Broker_ID"].fillna(-1).astype(int).astype(str)

    df["Acquisition_Channel"] = df["Acquisition_Channel"].fillna(
        df["Acquisition_Channel"].mode()[0]
    )

    df["Region_Code"] = df["Region_Code"].fillna("Unknown")

    df["Deductible_Tier"] = df["Deductible_Tier"].fillna(
        df["Deductible_Tier"].mode()[0]
    )

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
    df["Month_Num"] = df["Policy_Start_Month"].map(MONTH_ORDER)
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
    df["Deductible_Tier"] = df["Deductible_Tier"].map(DEDUCTIBLE_ORDER)

    # Binary: Broker_Agency_Type
    df["Is_National_Corporate"] = (df["Broker_Agency_Type"] == "National_Corporate").astype(int)
    df.drop(columns=["Broker_Agency_Type"], inplace=True)

    # ── SAFE One-Hot Encoding (manual) ──
    # Hardcode ALL known categories so train and test produce identical columns.
    ohe_specs = {
        "Acquisition_Channel": ["Broker_Referral", "Direct_Online", "Employer_Group", "Third_Party_Aggregator"],
        "Payment_Schedule":    ["Monthly", "Quarterly", "Semi_Annual"],
        "Employment_Status":   ["Part_Time", "Self_Employed", "Unemployed"],
    }
    for col, categories in ohe_specs.items():
        for cat in categories:
            new_col = f"{col}_{cat}"
            df[new_col] = (df[col] == cat).astype(int)
        df.drop(columns=[col], inplace=True)

    # Frequency encoding for Region_Code
    region_freq = df["Region_Code"].value_counts(normalize=True)
    df["Region_Code_FreqEnc"] = df["Region_Code"].map(region_freq).fillna(0.0)
    df.drop(columns=["Region_Code"], inplace=True)

    # Frequency encoding for Broker_ID
    broker_freq = df["Broker_ID"].value_counts(normalize=True)
    df["Broker_ID_FreqEnc"] = df["Broker_ID"].map(broker_freq).fillna(0.0)
    df.drop(columns=["Broker_ID"], inplace=True)

    return df


# ═══════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════
def load_model():
    """Load the trained model bundle from model.pkl."""
    return joblib.load("model.pkl")


# ═══════════════════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════════════════
def predict(df, model_bundle):
    """
    Takes a preprocessed DataFrame + trained model bundle.
    Returns a DataFrame with User_ID and Purchased_Coverage_Bundle
    (original string labels like 'Basic_Health').
    """
    # Unpack the bundle
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    inv_label_mapping = model_bundle["inv_label_mapping"]

    user_ids = df["User_ID"].reset_index(drop=True)

    # Drop non-feature columns
    features = df.drop(columns=["User_ID"], errors="ignore")
    for col in ["Purchased_Coverage_Bundle", "Policy_Cancelled_Post_Purchase"]:
        if col in features.columns:
            features = features.drop(columns=[col])

    # Align columns to exactly match training feature set
    # Add missing columns as 0, drop extra columns
    for col in feature_columns:
        if col not in features.columns:
            features[col] = 0
    features = features[feature_columns]

    # Predict integer labels, then map back to original strings
    preds = model.predict(features)
    labels = [inv_label_mapping[int(p)] for p in preds]

    result = pd.DataFrame({
        "User_ID": user_ids,
        "Purchased_Coverage_Bundle": labels
    })

    assert len(result) == len(user_ids), (
        f"Missing predictions! Expected {len(user_ids)}, got {len(result)}"
    )
    return result


# ═══════════════════════════════════════════════════════════
# TRAINING (only when executed directly)
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import f1_score
    from xgboost import XGBClassifier

    print("=" * 70)
    print("  INSURANCE BUNDLE RECOMMENDATION — TRAINING PIPELINE (FIXED)")
    print("=" * 70)

    # ─────────────────────────────────────
    # A. LOAD & PREPROCESS TRAIN DATA
    # ─────────────────────────────────────
    raw_train = pd.read_csv("train.csv")
    print(f"✅ Loaded train.csv: {raw_train.shape[0]:,} rows × {raw_train.shape[1]} columns")

    df = preprocess(raw_train)
    print(f"✅ Preprocessed: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ─────────────────────────────────────
    # B. PREPARE FEATURES & TARGET
    # ─────────────────────────────────────
    target_col = "Purchased_Coverage_Bundle"

    # Encode string labels → integers (sorted for determinism)
    sorted_labels = sorted(df[target_col].unique())
    label_mapping = {label: idx for idx, label in enumerate(sorted_labels)}
    inv_label_mapping = {idx: label for label, idx in label_mapping.items()}
    y = df[target_col].map(label_mapping)

    print(f"✅ Label mapping ({len(label_mapping)} classes):")
    for lbl, idx in label_mapping.items():
        print(f"   {idx} → {lbl}")

    drop_cols = ["User_ID", target_col, "Policy_Cancelled_Post_Purchase"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)

    feature_columns = list(X.columns)

    print(f"✅ Features: {X.shape[1]} | Target classes: {y.nunique()}")
    print(f"   Class distribution:\n{y.value_counts().sort_index()}")
    print("=" * 70)

    # ─────────────────────────────────────
    # C. TRAIN XGBOOST
    # ─────────────────────────────────────
    best_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )

    # Cross-validate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        best_model, X, y, cv=cv,
        scoring=["accuracy", "f1_macro"],
        n_jobs=-1
    )
    print(f"\n📊 XGBoost 5-Fold CV Results:")
    print(f"   Accuracy  : {cv_results['test_accuracy'].mean():.4f}")
    print(f"   F1 Macro  : {cv_results['test_f1_macro'].mean():.4f}")

    best_f1 = cv_results["test_f1_macro"].mean()

    # ─────────────────────────────────────
    # D. TRAIN ON FULL DATA & SAVE AS BUNDLE
    # ─────────────────────────────────────
    print(f"\n💾 Training XGBoost on FULL dataset ...")
    best_model.fit(X, y)

    # Save as a bundle with feature columns + label mapping for alignment
    model_bundle = {
        "model": best_model,
        "feature_columns": feature_columns,
        "label_mapping": label_mapping,
        "inv_label_mapping": inv_label_mapping,
    }
    joblib.dump(model_bundle, "model.pkl", compress=3)

    size_mb = os.path.getsize("model.pkl") / (1024 * 1024)
    print(f"✅ model.pkl saved ({size_mb:.2f} MB)")

    # ─────────────────────────────────────
    # E. GENERATE SUBMISSION ON TEST DATA
    # ─────────────────────────────────────
    print("\n📤 Generating submission from test.csv ...")
    raw_test = pd.read_csv("test.csv")
    test_df = preprocess(raw_test)

    loaded_bundle = load_model()
    submission = predict(test_df, loaded_bundle)
    submission.to_csv("submission.csv", index=False)
    print(f"✅ submission.csv saved ({submission.shape[0]:,} predictions)")

    # ─────────────────────────────────────
    # F. ESTIMATE FINAL HACKATHON SCORE
    # ─────────────────────────────────────
    raw_test_bench = pd.read_csv("test.csv")
    test_bench = preprocess(raw_test_bench)
    loaded = load_model()

    start_time = time.time()
    _ = predict(test_bench, loaded)
    latency_s = time.time() - start_time

    size_penalty = max(0.5, 1 - size_mb / 200)
    latency_penalty = max(0.5, 1 - latency_s / 10)
    final_score = best_f1 * size_penalty * latency_penalty

    print("\n" + "=" * 70)
    print("🏁 ESTIMATED FINAL HACKATHON SCORE")
    print("=" * 70)
    print(f"  Macro F1       : {best_f1:.4f}")
    print(f"  Model Size     : {size_mb:.2f} MB")
    print(f"  Latency        : {latency_s:.2f} s")
    print(f"  Size Penalty   : {size_penalty:.4f}")
    print(f"  Latency Penalty: {latency_penalty:.4f}")
    print(f"  🎯 Final Score : {final_score:.4f}")
    print("=" * 70)