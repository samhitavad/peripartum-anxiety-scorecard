"""
====================================================================
Caregiver Peripartum-Anxiety Risk Pipeline — Rural India Case Study
====================================================================
Implements the Methods sections 3.1-3.7 of the accompanying paper:

  3.2  Variable temporal layering  (L1 structural / L2 peripartum / L3 general)
  3.3  Cleaning + iterative imputation (chained equations)
  3.4  Profile  -- K-Prototypes typologies on Layer-1 structural variables
  3.5  Predict  -- 1-SE LASSO + 100x bootstrap stability selection
  3.6  Deploy   -- integer-point risk scorecard (PPI/Kessler-10-style)
  3.7  Robustness check against an independent second outcome measure

Run:  python pipeline.py <path_to_raw_qualtrics_export.xlsx> [output_dir]

All intermediate frames are written to <output_dir> as .pkl/.csv so any
single stage can be re-run or inspected without re-running the whole chain.
"""
from __future__ import annotations
import sys, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, confusion_matrix
from kmodes.kprototypes import KPrototypes
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
N_BOOTSTRAPS = 100

# --------------------------------------------------------------------
# Variable layers (Methods 3.2)
# --------------------------------------------------------------------
LAYER1 = ["Mother_Age_Pregnancy", "Father_Age_Pregnancy", "Mother_Education_Ord",
          "Father_Education_Ord", "Father_Employed_Bin", "Father_Income",
          "Family_Disability_Any_Bin", "Consanguinity_Bin", "Hazardous_Exposure_Bin"]
LAYER1_NOM = ["Religion"]
LAYER2 = ["Support_Spouse_Ord", "Support_Family_Ord", "Trauma_Score", "Obstetric_Loss_Bin",
          "Medications_Pregnancy_Bin", "Viral_Infection_Bin", "Gestational_Diab_Hypo_Bin",
          "Smoker_in_Home_Bin", "Vitamins_Bin", "Nutrition_Ord", "Water_Quality_Ord",
          "Mother_Worked_Pregnancy_Bin", "Premature_LBW_Bin", "Breastfed_Bin"]
LAYER3 = ["Father_Intoxication_Score", "GC_Awareness_Ord", "GC_View_Ord"]
OUTCOMES = ["Y_Peripartum_Anxiety", "Y_History_AnxDep"]
LAYER_TAG = {**{v: "L1" for v in LAYER1}, **{v: "L2" for v in LAYER2}, **{v: "L3" for v in LAYER3}}

RENAME = {
    "About The Surveyee": "Respondent_Type", "Type of Disability": "Disability_Type",
    "Sex": "Child_Sex", "Family History": "Family_Disability_Any", "Age": "Mother_Age_Pregnancy",
    "Education": "Mother_Education", "Religion": "Religion", "Location": "State_Pregnancy",
    "Birth History": "Obstetric_Loss", "Lifestyle": "Smoker_in_Home", "Lifestyle.1": "Vitamins",
    "Lifestyle.2": "Nutrition", "Child Birth": "Premature_LBW", "Child Birth.1": "Breastfed",
    "Lifestyle.3": "Water_Quality", "Lifestyle.4": "Mother_Worked_Pregnancy",
    "Mental Health": "Peripartum_Anxiety", "Mental Health.1": "History_AnxDep",
    "Mental Health.2": "Support_Spouse", "Mental Health.3": "Support_Family",
    "Physical Health": "Medications_Pregnancy", "Physical Health.1": "Viral_Infection",
    "Physical Health.2": "Gestational_Diab_Hypo", "Health": "Trauma",
    "Age.1": "Father_Age_Pregnancy", "Education.1": "Father_Education", "Marriage": "Consanguinity",
    "Employment": "Father_Employed", "Employment.3": "Father_Income", "Exposure": "Hazardous_Exposure",
    "Lifestyle.5": "Father_Intoxication", "Support": "Father_Financial_Support",
    "GC": "GC_Awareness", "GC.1": "GC_View",
}

EDU_MAP = {"Dropped out before 10th standard": 0, "Completed 10th standard": 1,
           "Completed 12th standard": 2, "Vocational Training": 3,
           "Attended college, did not graduate": 3, "Completed Bachelors Degree": 4,
           "Completed Masters Degree": 5}


# ============================================================
# STEP 1  --  Load, rename, encode, temporally tag  (3.2, 3.3)
# ============================================================
def multi_yesno(series, yes_tokens, no_tokens):
    """Qualtrics multi-select strings may contain internal commas within a
    single option (e.g. 'Yes, prior to pregnancy'), so options are matched
    by substring containment against the known vocabulary, not by naive
    comma-splitting."""
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan); continue
        has_yes = any(t in v for t in yes_tokens)
        has_no = any(t in v for t in no_tokens)
        out.append(1 if has_yes else 0 if has_no else np.nan)
    return pd.Series(out, index=series.index)


def recover_education(main_series, text_series):
    """Recover ordinal education level from free-text 'Other' responses
    ('5th', '8th standard', 'No education', 'Diploma') instead of discarding
    those rows to missingness."""
    out = main_series.map(EDU_MAP)
    below_10 = re.compile(r"^\s*(no|not|none|nil|na|illiterate|\d\s*(th|std))", re.I)
    diploma = re.compile(r"diploma|pgdca|dda|iti", re.I)
    twelfth = re.compile(r"12", re.I)
    for i in main_series.index:
        if main_series[i] == "Other (Type Below)" and (i not in out.index or pd.isna(out[i])):
            t = text_series[i]
            if pd.isna(t):
                continue
            t = str(t)
            if diploma.search(t): out[i] = 3
            elif twelfth.search(t): out[i] = 2
            elif below_10.search(t): out[i] = 0
    return out


def step1_load_and_encode(raw_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(raw_path)
    df = raw.iloc[1:].reset_index(drop=True).copy()  # row 0 is Qualtrics question text
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
    df = df.rename(columns=RENAME)
    raw_full = raw.iloc[1:].reset_index(drop=True)

    # --- Layer 1: structural ---
    df["Mother_Education_Ord"] = recover_education(df["Mother_Education"], raw_full["Education_8_TEXT"])
    df["Father_Education_Ord"] = recover_education(df["Father_Education"], raw_full["Education_8_TEXT.1"])
    df["Religion"] = df["Religion"].where(df["Religion"] != "Other", np.nan)
    df["Family_Disability_Any_Bin"] = df["Family_Disability_Any"].map(
        {"No, there are no other immediate family members with disabilities.": 0,
         "Yes, there are other immediate family members with disabilities": 1})
    df["Father_Employed_Bin"] = df["Father_Employed"].map({"Yes": 1, "No": 0})
    _emp_text = raw_full["Employment_3_TEXT"]
    _deceased = re.compile(r"expir|no more|expire|passed|death|died", re.I)
    for i in df.index:
        if df.loc[i, "Father_Employed"] == "Other" and pd.notna(_emp_text[i]):
            df.loc[i, "Father_Employed_Bin"] = 0 if _deceased.search(str(_emp_text[i])) else 1

    df["Father_Income"] = pd.to_numeric(df["Father_Income"], errors="coerce")
    n_low = int((df["Father_Income"] < 500).sum())
    n_high = int((df["Father_Income"] > 150000).sum())
    df.loc[df["Father_Income"] < 500, "Father_Income"] = np.nan
    df.loc[df["Father_Income"] > 150000, "Father_Income"] = np.nan
    print(f"[clean] Father_Income: flagged {n_low} implausibly-low and {n_high} implausibly-high "
          f"entries as missing (data-entry errors, not valid incomes)")

    df["Consanguinity_Bin"] = df["Consanguinity"].map({"Yes": 1, "No": 0})
    df["Hazardous_Exposure_Bin"] = df["Hazardous_Exposure"].map({"Yes": 1, "No": 0})
    df["Mother_Age_Pregnancy"] = pd.to_numeric(df["Mother_Age_Pregnancy"], errors="coerce")
    df["Father_Age_Pregnancy"] = pd.to_numeric(df["Father_Age_Pregnancy"], errors="coerce")

    # --- Layer 2: peripartum (explicitly "during her pregnancy") ---
    df["Obstetric_Loss_Bin"] = multi_yesno(df["Obstetric_Loss"], ["Yes, miscarriages", "Yes, still-births"], ["No"])
    df["Smoker_in_Home_Bin"] = df["Smoker_in_Home"].map({"Yes": 1, "No": 0})
    df["Vitamins_Bin"] = df["Vitamins"].map({"Yes": 1, "No": 0})
    NUTR_MAP = {"No, the mother did not have nutritious meals": 0,
                "Yes, the mother ate nutritious food 2-3 times a week": 1,
                "Yes, the mother ate nutritious food everyday": 2}
    df["Nutrition_Ord"] = df["Nutrition"].map(NUTR_MAP)
    df["Premature_LBW_Bin"] = multi_yesno(df["Premature_LBW"],
        ["Yes, Low Birth Weight", "Yes, Premature Birth"], ["No"])
    df["Breastfed_Bin"] = df["Breastfed"].map({"Yes": 1, "No": 0})
    WATER_MAP = {"Yes, she drank tap water": 0, "No, she drank filtered water": 1, "No, she drank mineral water": 2}
    df["Water_Quality_Ord"] = df["Water_Quality"].map(WATER_MAP)
    df["Mother_Worked_Pregnancy_Bin"] = multi_yesno(df["Mother_Worked_Pregnancy"],
        ["Yes, she worked before her pregnancy", "Yes, she worked during her pregnancy"], ["No, she did not work"])
    SUPP_SP = {"No emotional support from spouse": 0, "Yes, spouse provided support every now and then": 1,
               "Yes, spouse provided extremely adequate support": 2}
    SUPP_FAM = {"No emotional support from family": 0, "Yes, family provided support every now and then": 1,
                "Yes, family provided extremely adequate support": 2}
    df["Support_Spouse_Ord"] = df["Support_Spouse"].map(SUPP_SP)
    df["Support_Family_Ord"] = df["Support_Family"].map(SUPP_FAM)
    df["Medications_Pregnancy_Bin"] = df["Medications_Pregnancy"].map({"No.": 0, "Yes, she was taking medications": 1})
    df["Viral_Infection_Bin"] = df["Viral_Infection"].map({"No": 0, "Yes": 1})
    df["Gestational_Diab_Hypo_Bin"] = df["Gestational_Diab_Hypo"].map(
        {"No.": 0, "Yes, gestational diabetes": 1, "Yes, hypothyroidism": 1})

    def trauma_score(v):
        if pd.isna(v): return np.nan
        if v.startswith("No."): return 0
        n = sum(1 for t in ("Yes, emotional trauma", "Yes, physical trauma") if t in v)
        return min(n, 2) if n > 0 else np.nan
    df["Trauma_Score"] = df["Trauma"].apply(trauma_score)

    # --- Outcomes ---
    ANX_MAP = {"No": 0, "Yes, but only occasionally": 1, "Yes, these feelings were experienced often": 2}
    df["Peripartum_Anxiety_Ord"] = df["Peripartum_Anxiety"].map(ANX_MAP)
    df["Y_Peripartum_Anxiety"] = (df["Peripartum_Anxiety_Ord"] >= 1).astype("Int64")

    def histanxdep_score(v):
        if pd.isna(v): return np.nan
        has_yes = ("Yes, prior to pregnancy" in v) or ("Yes, post pregancy" in v)
        has_no = "No History" in v
        return 1 if has_yes else 0 if has_no else np.nan
    df["Y_History_AnxDep"] = df["History_AnxDep"].apply(histanxdep_score)

    # --- Layer 3: general / lifetime ---
    def intox_score(v):
        if pd.isna(v): return np.nan
        if v == "No History": return 0
        n = sum(1 for t in ("Yes, drinking history", "Yes, smoking history", "Yes, other recreational intoxicants") if t in v)
        return min(n, 2) if n > 0 else np.nan
    df["Father_Intoxication_Score"] = df["Father_Intoxication"].apply(intox_score)
    GC_AW = {"The parent does not know what genetic counseling is": 0,
             "The parent knows what genetic counseling is, but has not gone through it before": 1,
             "Yes, the parent has done genetic counseling": 2}
    GC_VIEW = {"Parent does not know what genetic counseling is": 0,
               "Negative View: Does not want to go through the process": 1,
               "Positive View: Thinks it can be helpful": 2}
    df["GC_Awareness_Ord"] = df["GC_Awareness"].map(GC_AW)
    df["GC_View_Ord"] = df["GC_View"].map(GC_VIEW)

    # disability flags (reporting only, excluded from clustering/LASSO)
    DISAB = {"ASD": r"autism", "Intellectual_Learning": r"intellectual|learning",
             "Down_Syndrome": r"down syndrome", "Cerebral_Palsy": r"cerebral"}
    src = df["Disability_Type"].fillna("").str.lower()
    for name, pat in DISAB.items():
        df[f"Disab_{name}"] = src.str.contains(pat, regex=True).astype(int)
    df["Father_Income_Band"] = pd.cut(df["Father_Income"], bins=[-1, 10000, 20000, 40000, 1e9],
                                       labels=["<10k", "10-20k", "20-40k", "40k+"])
    return df


# ============================================================
# STEP 2  --  Iterative imputation (chained equations)  (3.3)
# ============================================================
def step2_impute(df: pd.DataFrame) -> pd.DataFrame:
    all_num = LAYER1 + LAYER2 + LAYER3 + OUTCOMES
    work = df[all_num + LAYER1_NOM].copy()

    # Income is not missing-at-random for non-employed fathers -- it's
    # structurally zero. Fix it before the imputer ever sees the column.
    known_unemployed = work["Father_Employed_Bin"] == 0
    n_fixed = int((known_unemployed & work["Father_Income"].isna()).sum())
    work.loc[known_unemployed, "Father_Income"] = 0
    print(f"[impute] Father_Income fixed to 0 (not statistically imputed) for {n_fixed} known-unemployed fathers")

    nominal_dummies = {}
    for col in LAYER1_NOM:
        d = pd.get_dummies(work[col], prefix=col).astype(float)
        d.loc[work[col].isna(), :] = np.nan
        nominal_dummies[col] = list(d.columns)
        work = pd.concat([work.drop(columns=[col]), d], axis=1)

    work = work.apply(pd.to_numeric, errors="coerce").astype(float)
    n_missing = int(work.isna().sum().sum())

    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=25, random_state=RANDOM_STATE,
                                sample_posterior=False, skip_complete=True)
    imputed = pd.DataFrame(imputer.fit_transform(work), columns=work.columns, index=work.index)

    int_cols = [c for c in imputed.columns if c not in
                ("Mother_Age_Pregnancy", "Father_Age_Pregnancy", "Father_Income")]
    imputed[int_cols] = imputed[int_cols].round()
    imputed["Father_Employed_Bin"] = imputed["Father_Employed_Bin"].round().clip(0, 1)
    imputed["Father_Income"] = imputed["Father_Income"].clip(lower=0)
    # re-enforce post-imputation: imputed-non-employed => income forced to 0
    imputed.loc[imputed["Father_Employed_Bin"] == 0, "Father_Income"] = 0
    imputed["Father_Income"] = imputed["Father_Income"].round(-2)

    for col, dcols in nominal_dummies.items():
        imputed[col] = imputed[dcols].idxmax(axis=1).str.replace(f"{col}_", "", regex=False)
        imputed = imputed.drop(columns=dcols)

    print(f"[impute] {n_missing} missing cells filled via iterative imputation (chained equations, "
          f"BayesianRidge, 25 iterations)")
    print(f"[impute] Peripartum anxiety prevalence after imputation: {imputed['Y_Peripartum_Anxiety'].mean():.3f}")

    carry = df[["Disab_ASD", "Disab_Intellectual_Learning", "Disab_Down_Syndrome", "Disab_Cerebral_Palsy",
                "Respondent_Type", "Father_Income_Band", "Disability_Type"]]
    return pd.concat([imputed, carry], axis=1)


# ============================================================
# STEP 3  --  Profile: K-Prototypes typologies  (3.4)
# ============================================================
def step3_profile(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    X = df[LAYER1].astype(float).values
    cat = df[LAYER1_NOM].astype(str).values
    X_full = np.hstack([X, cat])
    cat_idx = list(range(len(LAYER1), len(LAYER1) + len(LAYER1_NOM)))

    print("[profile] K-Prototypes cost by k (elbow check):")
    for kk in [3, 4, 5, 6]:
        kp = KPrototypes(n_clusters=kk, init="Huang", n_init=10, random_state=RANDOM_STATE, verbose=0)
        kp.fit_predict(X_full, categorical=cat_idx)
        print(f"    k={kk}: cost={kp.cost_:.1f}")

    kp = KPrototypes(n_clusters=k, init="Huang", n_init=20, random_state=RANDOM_STATE, verbose=0)
    df["Cluster"] = kp.fit_predict(X_full, categorical=cat_idx)

    tbl = pd.crosstab(df["Cluster"], df["Y_Peripartum_Anxiety"])
    chi2, p, dof, _ = chi2_contingency(tbl)
    print(f"[profile] Omnibus chi-square (cluster x anxiety): chi2={chi2:.2f}, df={dof}, p={p:.4f}")
    ref = df["Cluster"].value_counts().idxmax()
    print(f"[profile] Per-cluster prevalence, 95% CI, and p-value vs. largest cluster (Cluster {ref}):")
    for c in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == c]
        n, kk_ = len(sub), int(sub["Y_Peripartum_Anxiety"].sum())
        lo, hi = proportion_confint(kk_, n, method="wilson")
        if c != ref:
            subref = df[df["Cluster"] == ref]
            _, pv = proportions_ztest([kk_, int(subref["Y_Peripartum_Anxiety"].sum())], [n, len(subref)])
        else:
            pv = np.nan
        print(f"    Cluster {c}: n={n:3d}  {kk_}/{n}={kk_/n*100:5.1f}%  95%CI=[{lo*100:.1f}%,{hi*100:.1f}%]  p_vs_ref={pv:.3f}" if pv==pv
              else f"    Cluster {c}: n={n:3d}  {kk_}/{n}={kk_/n*100:5.1f}%  95%CI=[{lo*100:.1f}%,{hi*100:.1f}%]  (reference)")
    return df


# ============================================================
# STEP 4  --  Predict: 1-SE LASSO  (3.5)
# ============================================================
def _candidate_matrix(df):
    religion_dum = pd.get_dummies(df["Religion"], prefix="Religion", drop_first=True).astype(float)
    for c in religion_dum.columns:
        LAYER_TAG.setdefault(c, "L1")
    X = pd.concat([df[LAYER1 + LAYER2 + LAYER3].astype(float), religion_dum], axis=1)
    return X


def fit_1se_lasso(X, y, seed=RANDOM_STATE, cv_folds=5, c_grid=None):
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    Cs = c_grid if c_grid is not None else np.logspace(-3, 2, 40)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cvmod = LogisticRegressionCV(Cs=Cs, cv=cv, penalty="l1", solver="liblinear",
                                  scoring="roc_auc", max_iter=5000, random_state=seed).fit(Xs, y)
    fold_scores = list(cvmod.scores_.values())[0]
    mean_auc = fold_scores.mean(axis=0); se_auc = fold_scores.std(axis=0) / np.sqrt(fold_scores.shape[0])
    best_idx = int(np.argmax(mean_auc)); thr = mean_auc[best_idx] - se_auc[best_idx]
    chosen_idx = int(np.where(mean_auc >= thr)[0].min())
    chosen_C = float(Cs[chosen_idx])
    final = LogisticRegression(penalty="l1", solver="liblinear", C=chosen_C,
                                max_iter=5000, random_state=seed).fit(Xs, y)
    return final, scaler, {"best_AUC": mean_auc[best_idx], "chosen_C": chosen_C, "chosen_AUC": mean_auc[chosen_idx]}


def coef_table(model, X, eps=1e-4):
    coef = np.where(np.abs(model.coef_.ravel()) < eps, 0.0, model.coef_.ravel())
    tbl = pd.DataFrame({"Variable": X.columns, "Coef_std": coef, "OR_per_1SD": np.exp(coef)})
    tbl["Layer"] = tbl["Variable"].map(LAYER_TAG)
    tbl = tbl[tbl["Coef_std"] != 0].copy()
    tbl["Abs"] = tbl["Coef_std"].abs()
    tbl = tbl.sort_values("Abs", ascending=False).drop(columns="Abs")
    tbl["Direction"] = np.where(tbl["Coef_std"] > 0, "higher risk", "protective")
    return tbl.round(4).reset_index(drop=True)


def step4_predict(df: pd.DataFrame):
    X = _candidate_matrix(df)
    results = {}
    for target in ["Y_Peripartum_Anxiety", "Y_History_AnxDep"]:
        y = df[target].astype(int).values
        model, scaler, info = fit_1se_lasso(X, y)
        tbl = coef_table(model, X)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        prob = cross_val_predict(LogisticRegression(penalty="l1", solver="liblinear", C=info["chosen_C"],
                                                      max_iter=5000, random_state=RANDOM_STATE),
                                  scaler.transform(X), y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, prob)
        print(f"[predict:{target}] CV AUC={auc:.3f}; survivors:")
        print(tbl.to_string(index=False))
        results[target] = {"table": tbl, "auc": auc, "model": model, "scaler": scaler}
    return X, results


# ============================================================
# STEP 5  --  Bootstrap stability selection  (3.5)
# ============================================================
def step5_bootstrap(X: pd.DataFrame, y: np.ndarray, n_boot=N_BOOTSTRAPS):
    feat = list(X.columns); Xnp = X.values; n = len(y)
    rng = np.random.default_rng(RANDOM_STATE)
    p = len(feat)
    sel = np.zeros(p, int); pos = np.zeros(p, int); neg = np.zeros(p, int); coefsum = np.zeros(p, float)
    ok = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, n); yb = y[idx]
        if yb.sum() < 5 or (1 - yb).sum() < 5:
            continue
        scaler = StandardScaler(); Xs = scaler.fit_transform(Xnp[idx, :])
        Cs = np.logspace(-2.5, 1.5, 15)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + b)
        try:
            cvm = LogisticRegressionCV(Cs=Cs, cv=cv, penalty="l1", solver="liblinear",
                                        scoring="roc_auc", max_iter=1000, random_state=RANDOM_STATE + b).fit(Xs, yb)
        except Exception:
            continue
        fs = list(cvm.scores_.values())[0]
        mean_auc = fs.mean(axis=0); se = fs.std(axis=0) / np.sqrt(fs.shape[0])
        bi = int(np.argmax(mean_auc)); thr = mean_auc[bi] - se[bi]
        ci = int(np.where(mean_auc >= thr)[0].min())
        final = LogisticRegression(penalty="l1", solver="liblinear", C=float(Cs[ci]),
                                    max_iter=1000, random_state=RANDOM_STATE + b).fit(Xs, yb)
        c = np.where(np.abs(final.coef_.ravel()) < 1e-4, 0.0, final.coef_.ravel())
        ok += 1; nz = c != 0
        sel += nz.astype(int); pos += (c > 0).astype(int); neg += (c < 0).astype(int)
        coefsum += np.where(nz, c, 0.0)

    freq = sel / max(ok, 1)
    out = pd.DataFrame({
        "Variable": feat, "Layer": [LAYER_TAG.get(f, "L1") for f in feat],
        "Selection_freq": freq.round(3),
        "Sign_consistency": np.where(sel > 0, np.maximum(pos, neg) / np.maximum(sel, 1), 0).round(3),
        "Mean_coef_when_selected": np.where(sel > 0, coefsum / np.maximum(sel, 1), 0).round(4),
        "Dominant_sign": np.where(pos >= neg, "+", "-"),
    }).sort_values("Selection_freq", ascending=False)
    out["Tier"] = np.select([out.Selection_freq >= 0.8, out.Selection_freq >= 0.6],
                             ["Very stable", "Stable"], default="Weaker evidence")
    print(f"[bootstrap] {ok}/{n_boot} successful resamples")
    print(out[out.Selection_freq > 0].to_string(index=False))
    return out


# ============================================================
# STEP 6  --  Deploy: integer-point risk scorecard  (3.6)
# ============================================================
def step6_scorecard(df: pd.DataFrame, boot: pd.DataFrame):
    very_stable = boot[boot["Selection_freq"] >= 0.80].copy()
    vars_ = very_stable["Variable"].tolist()
    raw_sd = df[vars_].astype(float).std()
    raw_coef = very_stable.set_index("Variable")["Mean_coef_when_selected"] / raw_sd
    base_unit = raw_coef.abs().min()
    points = (raw_coef / base_unit * 3).round().astype(int)
    points = points.where(points.abs() >= 1, np.sign(points).replace(0, 1))

    print("[scorecard] Point values:")
    for v in vars_:
        prevalence = df[v].mean()
        print(f"    {v:28s} {points[v]:+d} pts/unit   (sample prevalence/mean={prevalence:.3f})")
    print("[scorecard] NOTE: points are per raw-unit increase, derived by rescaling each "
          "variable's standardized coefficient by its own raw SD. Rarer binary items (e.g. "
          "infection, prevalence ~7%) receive a larger per-unit weight than more common items "
          "with similar standardized coefficients, because their raw SD is small. This is a "
          "disclosed design choice, not an error: rare-but-stable predictors carry more "
          "information per occurrence, but it should be stated explicitly when the scorecard "
          "is presented.")

    df["Risk_Score"] = sum(df[v].astype(float) * points[v] for v in vars_)
    y = df["Y_Peripartum_Anxiety"].astype(int).values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs = [roc_auc_score(y[te], df["Risk_Score"].values[te]) for _, te in cv.split(df, y)]
    print(f"[scorecard] 5-fold CV AUC of fixed point total: {np.mean(aucs):.3f} (+/- {np.std(aucs):.3f})")
    print("[scorecard] CAVEAT: variable selection + weight estimation used the full sample; "
          "this AUC tests the fixed scorecard on held-out households but is not a fully nested "
          "estimate of the whole model-building process, and may be modestly optimistic.")

    oof_bin = pd.Series(index=df.index, dtype=float)
    for tr, te in cv.split(df, y):
        edges = np.quantile(df["Risk_Score"].values[tr], [0, .2, .4, .6, .8, 1.0])
        edges[0] -= 1; edges[-1] += 1
        oof_bin.iloc[te] = pd.cut(df["Risk_Score"].values[te], bins=edges, labels=False, include_lowest=True)
    calib = pd.DataFrame({"quintile": oof_bin, "y": y, "score": df["Risk_Score"].values})
    calib_tbl = calib.groupby("quintile").agg(N=("y", "size"), min_score=("score", "min"),
                                               max_score=("score", "max"), prevalence=("y", "mean"))
    calib_tbl["prevalence_%"] = (calib_tbl["prevalence"] * 100).round(1)
    print("[scorecard] Out-of-fold calibration table:")
    print(calib_tbl[["N", "min_score", "max_score", "prevalence_%"]].to_string())
    return df, points, calib_tbl


# ============================================================
# STEP 7  --  Robustness check  (3.7)
# ============================================================
def step7_robustness(df: pd.DataFrame, results: dict):
    y_alt = df["Y_History_AnxDep"].astype(int).values
    y_primary = df["Y_Peripartum_Anxiety"].astype(int).values
    auc_alt = roc_auc_score(y_alt, df["Risk_Score"].values)
    auc_primary = roc_auc_score(y_primary, df["Risk_Score"].values)
    print(f"[robustness] Fixed scorecard AUC vs. PRIMARY outcome (in-sample): {auc_primary:.3f}")
    print(f"[robustness] Fixed scorecard AUC vs. ALTERNATE outcome (never used to build it): {auc_alt:.3f}")
    shared = set(results["Y_Peripartum_Anxiety"]["table"]["Variable"]) & \
             set(results["Y_History_AnxDep"]["table"]["Variable"])
    print(f"[robustness] Variables surviving single-shot LASSO under BOTH outcomes: {shared}")


# ============================================================
# ORCHESTRATOR
# ============================================================
def run_pipeline(raw_path: str, out_dir: str = "outputs"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70); print("STEP 1 -- Load, clean, temporally tag"); print("=" * 70)
    df = step1_load_and_encode(Path(raw_path))
    df.to_pickle(out_dir / "step1_encoded.pkl")

    print("\n" + "=" * 70); print("STEP 2 -- Iterative imputation"); print("=" * 70)
    df = step2_impute(df)
    df.to_pickle(out_dir / "step2_imputed.pkl")

    print("\n" + "=" * 70); print("STEP 3 -- Profile: K-Prototypes typologies"); print("=" * 70)
    df = step3_profile(df)
    df.to_pickle(out_dir / "step3_profiled.pkl")

    print("\n" + "=" * 70); print("STEP 4 -- Predict: 1-SE LASSO (primary + alternate outcome)"); print("=" * 70)
    X, results = step4_predict(df)
    results["Y_Peripartum_Anxiety"]["table"].to_csv(out_dir / "lasso_primary.csv", index=False)
    results["Y_History_AnxDep"]["table"].to_csv(out_dir / "lasso_robustness.csv", index=False)

    print("\n" + "=" * 70); print("STEP 5 -- Bootstrap stability selection (100x)"); print("=" * 70)
    y_primary = df["Y_Peripartum_Anxiety"].astype(int).values
    boot = step5_bootstrap(X, y_primary)
    boot.to_csv(out_dir / "bootstrap_stability.csv", index=False)

    print("\n" + "=" * 70); print("STEP 6 -- Deploy: risk scorecard + calibration"); print("=" * 70)
    df, points, calib_tbl = step6_scorecard(df, boot)
    df.to_pickle(out_dir / "step6_scored.pkl")
    calib_tbl.to_csv(out_dir / "calibration_table.csv")

    print("\n" + "=" * 70); print("STEP 7 -- Robustness check vs. alternate outcome"); print("=" * 70)
    step7_robustness(df, results)

    print(f"\nAll artifacts written to: {out_dir.resolve()}")


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "Cleaned_Survey_Data.xlsx"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    run_pipeline(raw, outdir)
