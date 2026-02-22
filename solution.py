import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")


# ── Constants ──

LABEL_TO_INT = {
    "Auto_Comprehensive": 0, "Auto_Liability_Basic": 1,
    "Basic_Health": 2, "Family_Comprehensive": 3,
    "Health_Dental_Vision": 4, "Home_Premium": 5,
    "Home_Standard": 6, "Premium_Health_Life": 7,
    "Renter_Basic": 8, "Renter_Premium": 9,
}

MONTH_ORDER = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

DEDUCTIBLE_ORDER = {
    "Tier_1_High_Ded": 1, "Tier_2_Mid_Ded": 2,
    "Tier_3_Low_Ded": 3, "Tier_4_Zero_Ded": 4,
}

OHE_SPECS = {
    "Acquisition_Channel": ["Broker_Referral", "Direct_Online", "Employer_Group", "Third_Party_Aggregator"],
    "Payment_Schedule": ["Monthly", "Quarterly", "Semi_Annual"],
    "Employment_Status": ["Part_Time", "Self_Employed", "Unemployed"],
}


# ═══════════════════════════════════════════════════════════
#  preprocess(df)  —  Returns a pandas DataFrame
# ═══════════════════════════════════════════════════════════
def preprocess(df):
    df = df.copy()

    df["Has_Employer_ID"] = df["Employer_ID"].notna().astype(int)
    df.drop(columns=["Employer_ID"], inplace=True)

    df["Has_Broker_ID"] = df["Broker_ID"].notna().astype(int)
    df["Broker_ID"] = df["Broker_ID"].fillna(-1).astype(int).astype(str)
    df["Acquisition_Channel"] = df["Acquisition_Channel"].fillna(df["Acquisition_Channel"].mode()[0])
    df["Region_Code"] = df["Region_Code"].fillna("Unknown")
    df["Deductible_Tier"] = df["Deductible_Tier"].fillna(df["Deductible_Tier"].mode()[0])
    df["Child_Dependents"] = df["Child_Dependents"].fillna(df["Child_Dependents"].median()).astype(int)

    df["Child_Dependents"] = df["Child_Dependents"].clip(upper=5)
    income_cap = df["Estimated_Annual_Income"].quantile(0.99)
    df["Estimated_Annual_Income"] = df["Estimated_Annual_Income"].clip(upper=income_cap)
    df["Estimated_Annual_Income_Log"] = np.log1p(df["Estimated_Annual_Income"])
    df.drop(columns=["Estimated_Annual_Income"], inplace=True)
    days_cap = df["Days_Since_Quote"].quantile(0.99)
    df["Days_Since_Quote"] = df["Days_Since_Quote"].clip(upper=days_cap)
    df["Underwriting_Processing_Days"] = np.log1p(df["Underwriting_Processing_Days"])

    df["Total_Dependents"] = df["Adult_Dependents"] + df["Child_Dependents"] + df["Infant_Dependents"]

    df["Month_Num"] = df["Policy_Start_Month"].map(MONTH_ORDER)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month_Num"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month_Num"] / 12)
    df.drop(columns=["Policy_Start_Month", "Month_Num"], inplace=True)
    df["Day_Sin"] = np.sin(2 * np.pi * df["Policy_Start_Day"] / 31)
    df["Day_Cos"] = np.cos(2 * np.pi * df["Policy_Start_Day"] / 31)
    df.drop(columns=["Policy_Start_Day"], inplace=True)
    df["Week_Sin"] = np.sin(2 * np.pi * df["Policy_Start_Week"] / 52)
    df["Week_Cos"] = np.cos(2 * np.pi * df["Policy_Start_Week"] / 52)
    df.drop(columns=["Policy_Start_Week"], inplace=True)

    df["Deductible_Tier"] = df["Deductible_Tier"].map(DEDUCTIBLE_ORDER)
    df["Is_National_Corporate"] = (df["Broker_Agency_Type"] == "National_Corporate").astype(int)
    df.drop(columns=["Broker_Agency_Type"], inplace=True)

    for col, cats in OHE_SPECS.items():
        for cat in cats:
            df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
        df.drop(columns=[col], inplace=True)

    region_freq = df["Region_Code"].value_counts(normalize=True)
    df["Region_Code_Freq"] = df["Region_Code"].map(region_freq).fillna(0.0)
    df.drop(columns=["Region_Code"], inplace=True)

    broker_freq = df["Broker_ID"].value_counts(normalize=True)
    df["Broker_ID_Freq"] = df["Broker_ID"].map(broker_freq).fillna(0.0)
    df.drop(columns=["Broker_ID"], inplace=True)

    return df


# ═══════════════════════════════════════════════════════════
#  load_model()  —  Returns your loaded model object
# ═══════════════════════════════════════════════════════════
def load_model():
    return joblib.load("model.pkl")


# ═══════════════════════════════════════════════════════════
#  predict(df, model)  —  Returns a DataFrame with User_ID
#                         and Purchased_Coverage_Bundle
# ═══════════════════════════════════════════════════════════
def predict(df, model):
    xgb_model = model["model"]
    feature_cols = model["feature_columns"]

    user_ids = df["User_ID"].reset_index(drop=True)

    features = df.drop(columns=["User_ID"], errors="ignore")
    for col in ["Purchased_Coverage_Bundle", "Policy_Cancelled_Post_Purchase"]:
        if col in features.columns:
            features = features.drop(columns=[col])

    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0
    features = features[feature_cols]

    preds = xgb_model.predict(features)

    return pd.DataFrame({
        "User_ID": user_ids,
        "Purchased_Coverage_Bundle": preds.astype(int),
    })


# ═══════════════════════════════════════════════════════════
#  Training  (only runs when executed directly)
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, time
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from xgboost import XGBClassifier

    raw = pd.read_csv("train.csv")
    df = preprocess(raw)

    y = df["Purchased_Coverage_Bundle"].map(LABEL_TO_INT)
    drop = [c for c in ["User_ID", "Purchased_Coverage_Bundle", "Policy_Cancelled_Post_Purchase"] if c in df.columns]
    X = df.drop(columns=drop)
    feature_columns = list(X.columns)

    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42, verbosity=0, n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_res = cross_validate(clf, X, y, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=-1)
    print(f"CV Accuracy: {cv_res['test_accuracy'].mean():.4f}")
    print(f"CV F1 Macro: {cv_res['test_f1_macro'].mean():.4f}")

    clf.fit(X, y)
    joblib.dump({"model": clf, "feature_columns": feature_columns}, "model.pkl", compress=3)
    print(f"model.pkl saved ({os.path.getsize('model.pkl') / 1024 / 1024:.2f} MB)")

    test_df = preprocess(pd.read_csv("test.csv"))
    sub = predict(test_df, load_model())
    sub.to_csv("submission.csv", index=False)
    print(f"submission.csv saved ({len(sub)} rows)")
    print(sub.head())
