import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

LABEL_TO_INT = {
    "Auto_Comprehensive": 0, "Auto_Liability_Basic": 1,
    "Basic_Health": 2, "Family_Comprehensive": 3,
    "Health_Dental_Vision": 4, "Home_Premium": 5,
    "Home_Standard": 6, "Premium_Health_Life": 7,
    "Renter_Basic": 8, "Renter_Premium": 9,
}

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

DEDUCTIBLE_MAP = {
    "Tier_1_High_Ded": 1, "Tier_2_Mid_Ded": 2,
    "Tier_3_Low_Ded": 3, "Tier_4_Zero_Ded": 4,
}

OHE_SPECS = {
    "Acquisition_Channel": ["Corporate_Partner", "Direct_Website", "Local_Broker", "Affiliate_Group"],
    "Payment_Schedule": ["Annual_Upfront", "Quarterly_Invoice"],
    "Employment_Status": ["Self_Employed", "Contractor", "Unemployed"],
}

_PREP_CACHE = None


def _load_prep():
    global _PREP_CACHE
    if _PREP_CACHE is None:
        try:
            _PREP_CACHE = joblib.load("model.pkl").get("prep", {})
        except Exception:
            _PREP_CACHE = {}
    return _PREP_CACHE


# ═══════════════════════════════════════════════════════════
#  preprocess(df)  —  Returns a pandas DataFrame
# ═══════════════════════════════════════════════════════════
def preprocess(df):
    df = df.copy()
    is_train = "Purchased_Coverage_Bundle" in df.columns
    art = {} if is_train else _load_prep()

    df["Has_Employer_ID"] = df["Employer_ID"].notna().astype(int)
    df.drop(columns=["Employer_ID"], inplace=True)

    df["Has_Broker_ID"] = df["Broker_ID"].notna().astype(int)
    df["Broker_ID"] = df["Broker_ID"].fillna(-1).astype(int).astype(str)
    df["Acquisition_Channel"] = df["Acquisition_Channel"].fillna(art.get("acq_mode", df["Acquisition_Channel"].mode()[0]))
    df["Region_Code"] = df["Region_Code"].fillna("Unknown")
    df["Deductible_Tier"] = df["Deductible_Tier"].fillna(art.get("ded_mode", df["Deductible_Tier"].mode()[0]))
    df["Child_Dependents"] = df["Child_Dependents"].fillna(art.get("child_med", df["Child_Dependents"].median())).astype(int)

    df["Child_Dependents"] = df["Child_Dependents"].clip(upper=5)
    inc_cap = art.get("income_cap", df["Estimated_Annual_Income"].quantile(0.99))
    df["Estimated_Annual_Income"] = df["Estimated_Annual_Income"].clip(upper=inc_cap)
    df["Log_Income"] = np.log1p(df["Estimated_Annual_Income"])
    df.drop(columns=["Estimated_Annual_Income"], inplace=True)

    day_cap = art.get("days_cap", df["Days_Since_Quote"].quantile(0.99))
    df["Days_Since_Quote"] = df["Days_Since_Quote"].clip(upper=day_cap)
    df["Log_UW_Days"] = np.log1p(df["Underwriting_Processing_Days"])
    df.drop(columns=["Underwriting_Processing_Days"], inplace=True)

    df["Total_Dependents"] = df["Adult_Dependents"] + df["Child_Dependents"] + df["Infant_Dependents"]
    df["Has_Dependents"] = (df["Total_Dependents"] > 0).astype(int)
    df["Has_Infant"] = (df["Infant_Dependents"] > 0).astype(int)
    df["Income_Per_Dependent"] = df["Log_Income"] / (df["Total_Dependents"] + 1)
    df["Claims_Ratio"] = df["Previous_Claims_Filed"] / (df["Years_Without_Claims"] + 1)
    df["Policy_Activity"] = df["Policy_Amendments_Count"] + df["Custom_Riders_Requested"]
    df["Has_Vehicle"] = (df["Vehicles_on_Policy"] > 0).astype(int)
    df["Multi_Vehicle"] = (df["Vehicles_on_Policy"] > 1).astype(int)
    df["Has_Riders"] = (df["Custom_Riders_Requested"] > 0).astype(int)
    df["Has_Grace_Ext"] = (df["Grace_Period_Extensions"] > 0).astype(int)
    df["Long_Policy"] = (df["Previous_Policy_Duration_Months"] > 12).astype(int)
    df["Risk_Score"] = df["Previous_Claims_Filed"] * 2 - df["Years_Without_Claims"]
    df["Quote_Freshness"] = 1.0 / (df["Days_Since_Quote"] + 1)

    mn = df["Policy_Start_Month"].map(MONTH_MAP)
    df["Month_Sin"] = np.sin(2 * np.pi * mn / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * mn / 12)
    df.drop(columns=["Policy_Start_Month"], inplace=True)
    df["Day_Sin"] = np.sin(2 * np.pi * df["Policy_Start_Day"] / 31)
    df["Day_Cos"] = np.cos(2 * np.pi * df["Policy_Start_Day"] / 31)
    df.drop(columns=["Policy_Start_Day"], inplace=True)
    df["Week_Sin"] = np.sin(2 * np.pi * df["Policy_Start_Week"] / 52)
    df["Week_Cos"] = np.cos(2 * np.pi * df["Policy_Start_Week"] / 52)
    df.drop(columns=["Policy_Start_Week"], inplace=True)

    df["Deductible_Tier"] = df["Deductible_Tier"].map(DEDUCTIBLE_MAP)
    df["Is_National_Corporate"] = (df["Broker_Agency_Type"] == "National_Corporate").astype(int)
    df.drop(columns=["Broker_Agency_Type"], inplace=True)

    for col, cats in OHE_SPECS.items():
        for cat in cats:
            df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
        df.drop(columns=[col], inplace=True)

    if "region_freq" in art:
        df["Region_Code_Freq"] = df["Region_Code"].map(art["region_freq"]).fillna(0.0)
    else:
        rf = df["Region_Code"].value_counts(normalize=True)
        df["Region_Code_Freq"] = df["Region_Code"].map(rf).fillna(0.0)

    if "broker_freq" in art:
        df["Broker_ID_Freq"] = df["Broker_ID"].map(art["broker_freq"]).fillna(0.0)
    else:
        bf = df["Broker_ID"].value_counts(normalize=True)
        df["Broker_ID_Freq"] = df["Broker_ID"].map(bf).fillna(0.0)

    df.drop(columns=["Region_Code", "Broker_ID"], inplace=True)
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
    feat_cols = model["feature_columns"]
    uids = df["User_ID"].values

    features = df.drop(columns=["User_ID"], errors="ignore")
    for c in ["Purchased_Coverage_Bundle", "Policy_Cancelled_Post_Purchase"]:
        if c in features.columns:
            features = features.drop(columns=[c])
    for c in feat_cols:
        if c not in features.columns:
            features[c] = 0
    X = features[feat_cols].values.astype(np.float32)

    models = model["models"]
    weights = model["weights"]

    # Weighted average of probabilities
    proba = None
    for m, w in zip(models, weights):
        p = m.predict_proba(X)
        proba = p * w if proba is None else proba + p * w
    preds = np.argmax(proba, axis=1)

    return pd.DataFrame({
        "User_ID": uids,
        "Purchased_Coverage_Bundle": preds.astype(int),
    })


# ═══════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, time
    from collections import Counter
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    raw = pd.read_csv("train.csv")
    print(f"Loaded: {raw.shape}")

    prep_art = {
        "acq_mode": raw["Acquisition_Channel"].mode()[0],
        "ded_mode": raw["Deductible_Tier"].mode()[0],
        "child_med": float(raw["Child_Dependents"].median()),
        "income_cap": float(raw["Estimated_Annual_Income"].quantile(0.99)),
        "days_cap": float(raw["Days_Since_Quote"].quantile(0.99)),
        "region_freq": raw["Region_Code"].fillna("Unknown").value_counts(normalize=True).to_dict(),
        "broker_freq": raw["Broker_ID"].fillna(-1).astype(int).astype(str).value_counts(normalize=True).to_dict(),
    }

    df = preprocess(raw)
    y = df["Purchased_Coverage_Bundle"].map(LABEL_TO_INT)
    drop = [c for c in ["User_ID", "Purchased_Coverage_Bundle", "Policy_Cancelled_Post_Purchase"] if c in df.columns]
    X = df.drop(columns=drop)
    feat_cols = list(X.columns)
    print(f"Features: {len(feat_cols)}, Classes: {y.nunique()}")

    counts = Counter(y)
    n, n_cls = len(y), len(counts)
    cw = {c: n / (n_cls * cnt) for c, cnt in counts.items()}
    sw = np.array([cw[yi] for yi in y])

    # ── Define candidate models ──
    candidates = {
        "LGB_A": LGBMClassifier(
            n_estimators=500, max_depth=7, learning_rate=0.05,
            num_leaves=50, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=30, reg_alpha=0.05, reg_lambda=0.5,
            is_unbalance=True, random_state=42, verbosity=-1, n_jobs=-1,
        ),
        "LGB_B": LGBMClassifier(
            n_estimators=800, max_depth=-1, learning_rate=0.02,
            num_leaves=31, subsample=0.7, colsample_bytree=0.7,
            min_child_samples=50, reg_alpha=0.1, reg_lambda=1.0,
            is_unbalance=True, random_state=42, verbosity=-1, n_jobs=-1,
        ),
        "XGB_A": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="mlogloss", random_state=42, verbosity=0, n_jobs=-1,
        ),
        "XGB_B": XGBClassifier(
            n_estimators=600, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            gamma=0.2, reg_alpha=0.2, reg_lambda=2.0,
            eval_metric="mlogloss", random_state=42, verbosity=0, n_jobs=-1,
        ),
    }

    # ── OOF predictions ──
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = {name: np.zeros((len(y), n_cls)) for name in candidates}

    for fold, (tr, va) in enumerate(cv.split(X, y)):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        sw_tr = sw[tr]

        for name, mdl in candidates.items():
            m = mdl.__class__(**mdl.get_params())
            m.fit(Xtr, ytr, sample_weight=sw_tr)
            oof[name][va] = m.predict_proba(Xva)

        scores = {name: f1_score(yva, np.argmax(oof[name][va], axis=1), average="macro") for name in candidates}
        print(f"Fold {fold}: " + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    # Individual scores
    print("\nIndividual OOF F1:")
    indiv = {}
    for name in candidates:
        f = f1_score(y, np.argmax(oof[name], axis=1), average="macro")
        indiv[name] = f
        print(f"  {name}: {f:.4f}")

    # ── Greedy ensemble search ──
    print("\nSearching best ensemble...")
    names = list(candidates.keys())
    best_ens_f1 = 0
    best_ens_w = None
    best_ens_names = None

    # Try all pairs and triples with weight grid
    from itertools import combinations
    for r in range(1, len(names) + 1):
        for combo in combinations(range(len(names)), r):
            if r == 1:
                w_options = [(1.0,)]
            elif r == 2:
                w_options = [(w, 1 - w) for w in np.arange(0.2, 0.85, 0.05)]
            elif r == 3:
                w_options = []
                for w1 in np.arange(0.1, 0.8, 0.1):
                    for w2 in np.arange(0.1, 0.9 - w1, 0.1):
                        w_options.append((w1, w2, round(1 - w1 - w2, 2)))
            else:
                w_options = [(1/r,) * r]

            for ws in w_options:
                p = sum(ws[i] * oof[names[combo[i]]] for i in range(len(combo)))
                f = f1_score(y, np.argmax(p, axis=1), average="macro")
                if f > best_ens_f1:
                    best_ens_f1 = f
                    best_ens_w = ws
                    best_ens_names = [names[c] for c in combo]

    print(f"Best ensemble: {best_ens_names} weights={best_ens_w} F1={best_ens_f1:.4f}")

    # ── Train final models ──
    print("\nTraining final models...")
    final_models = []
    final_weights = []
    for name, w in zip(best_ens_names, best_ens_w):
        mdl = candidates[name].__class__(**candidates[name].get_params())
        mdl.fit(X, y, sample_weight=sw)
        final_models.append(mdl)
        final_weights.append(w)
        print(f"  {name} (w={w:.2f}) trained")

    bundle = {
        "models": final_models,
        "weights": final_weights,
        "feature_columns": feat_cols,
        "prep": prep_art,
    }
    joblib.dump(bundle, "model.pkl", compress=3)
    size_mb = os.path.getsize("model.pkl") / (1024 * 1024)
    print(f"model.pkl: {size_mb:.2f} MB")

    # ── Verify ──
    _PREP_CACHE = None
    _PREP_CACHE = prep_art
    test_df = preprocess(pd.read_csv("test.csv"))
    loaded = load_model()

    t0 = time.time()
    sub = predict(test_df, loaded)
    lat = time.time() - t0

    sub.to_csv("submission.csv", index=False)
    print(f"\nsubmission.csv: {len(sub)} rows, latency={lat:.3f}s")
    print(sub["Purchased_Coverage_Bundle"].value_counts().sort_index())

    sp = max(0.5, 1 - size_mb / 200)
    lp = max(0.5, 1 - lat / 10)
    print(f"\nF1={best_ens_f1:.4f}  Size={size_mb:.2f}MB  Latency={lat:.3f}s")
    print(f"SizePen={sp:.4f}  LatPen={lp:.4f}")
    print(f"Estimated Score: {best_ens_f1 * sp * lp:.4f}")
