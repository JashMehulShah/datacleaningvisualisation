"""
Data Cleaning & Visualization Project
Dataset: raw retail-bank loan applications (messy, simulated)
Steps: 1) load raw data  2) audit  3) clean (duplicates, missing, outliers, types)
       4) visualize  5) key findings
Run: python clean_viz.py            (uses built-in raw dataset)
     python clean_viz.py my.csv     (use your own CSV; column names must match
                                     or edit the CONFIG section)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# 1. RAW DATA (generate a messy dataset if no CSV is passed)
# ---------------------------------------------------------------
def make_raw(n=1500):
    cities = ["Mumbai", "Delhi", "Hyderabad", "Bengaluru", "Chennai", "Pune"]
    df = pd.DataFrame({
        "customer_id": np.arange(1000, 1000 + n),
        "age": rng.normal(38, 11, n).clip(21, 70).round(),
        "city": rng.choice(cities, n),
        "employment": rng.choice(["Salaried", "Self-employed", "Business", "Student"], n, p=[.55, .2, .2, .05]),
        "annual_income": rng.lognormal(13.2, 0.45, n).round(),
        "credit_score": rng.normal(690, 70, n).clip(300, 900).round(),
        "loan_amount": rng.lognormal(12.6, 0.6, n).round(),
        "loan_term_months": rng.choice([12, 24, 36, 60, 84], n),
    })
    ratio = df.loan_amount / df.annual_income
    risk = 2.5 * (ratio - ratio.median()) + 1.2 * (690 - df.credit_score) / 70
    prob = 1 / (1 + np.exp(-(risk - 1.7)))
    df["default"] = (rng.random(n) < prob).astype(int)
    df["default"] = df["default"].map({1: "Yes", 0: "No"})

    # --- inject mess ---
    for col, frac in [("age", .06), ("annual_income", .08), ("credit_score", .05), ("city", .04), ("employment", .03)]:
        df.loc[df.sample(frac=frac, random_state=rng.integers(1e6)).index, col] = np.nan
    out_i = df.sample(10, random_state=1).index
    df.loc[out_i, "annual_income"] = df.loc[out_i, "annual_income"] * 25      # absurd incomes
    df.loc[df.sample(6, random_state=2).index, "age"] = [150, -4, 0, 999, 130, 210]  # impossible ages
    df.loc[df.sample(8, random_state=3).index, "loan_amount"] *= 30           # data-entry errors
    df["city"] = df["city"].replace({"Mumbai": rng.choice(["Mumbai", "mumbai ", "MUMBAI"])})  # inconsistent text
    df.loc[df.sample(40, random_state=4).index, "city"] = " hyderabad"
    df["annual_income"] = df["annual_income"].astype("object")
    df.loc[df.sample(25, random_state=5).index, "annual_income"] = "N/A"    # text in numeric col
    df = pd.concat([df, df.sample(60, random_state=6)], ignore_index=True)    # duplicates
    return df.sample(frac=1, random_state=7).reset_index(drop=True)

raw = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else make_raw()
raw.to_csv("raw_data.csv", index=False)
df = raw.copy()

# ---------------------------------------------------------------
# 2. AUDIT
# ---------------------------------------------------------------
log = []
def note(msg):
    print(msg); log.append(msg)

note(f"Raw shape: {df.shape}")
note(f"Duplicate rows: {df.duplicated().sum()}")
note("Missing values per column (raw):\n" + df.isna().sum().to_string())

# ---------------------------------------------------------------
# 3. CLEANING
# ---------------------------------------------------------------
# 3a. duplicates
before = len(df)
df = df.drop_duplicates()
note(f"\n[Duplicates] removed {before - len(df)} rows")

# 3b. fix types: text like 'N/A' in numeric columns -> NaN
for c in ["age", "annual_income", "credit_score", "loan_amount", "loan_term_months"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 3c. standardise text
for c in ["city", "employment", "default"]:
    df[c] = df[c].astype("string").str.strip().str.title()

# 3d. impossible values -> NaN (domain rules)
bad_age = ~df["age"].between(18, 80) & df["age"].notna()
note(f"[Invalid] ages outside 18-80 set to missing: {bad_age.sum()}")
df.loc[bad_age, "age"] = np.nan

# 3e. outliers: IQR rule, capped (winsorised) instead of dropped
def cap_iqr(s):
    q1, q3 = s.quantile([.25, .75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    n = ((s < lo) | (s > hi)).sum()
    return s.clip(lo, hi), n

for c in ["annual_income", "loan_amount"]:
    df[c], n = cap_iqr(df[c])
    note(f"[Outliers] {c}: {n} values capped using IQR rule")

# 3f. missing values
for c in ["age", "annual_income", "credit_score"]:     # numeric -> median
    note(f"[Missing] {c}: {df[c].isna().sum()} filled with median ({df[c].median():.0f})")
    df[c] = df[c].fillna(df[c].median())
for c in ["city", "employment"]:                       # categorical -> mode
    note(f"[Missing] {c}: {df[c].isna().sum()} filled with mode ({df[c].mode()[0]})")
    df[c] = df[c].fillna(df[c].mode()[0])

# 3g. feature engineering
df["loan_to_income"] = (df["loan_amount"] / df["annual_income"]).round(3)
df["age_group"] = pd.cut(df["age"], [17, 25, 35, 45, 55, 80], labels=["18-25", "26-35", "36-45", "46-55", "56+"])
df["score_band"] = pd.cut(df["credit_score"], [0, 600, 700, 750, 900], labels=["<600", "600-700", "700-750", "750+"])
df["is_default"] = (df["default"] == "Yes").astype(int)

assert df.isna().sum().sum() == 0 and not df.duplicated().any()
note(f"\nClean shape: {df.shape}  | missing: 0 | duplicates: 0")
df.to_csv("clean_data.csv", index=False)

# ---------------------------------------------------------------
# 4. VISUALIZATION
# ---------------------------------------------------------------
# before/after cleaning (outlier effect)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
pd.to_numeric(raw["loan_amount"], errors="coerce").plot.box(ax=ax[0], vert=False)
ax[0].set_title("Loan amount - BEFORE (outliers)")
df["loan_amount"].plot.box(ax=ax[1], vert=False, color="green")
ax[1].set_title("Loan amount - AFTER (capped)")
plt.tight_layout(); plt.savefig("fig_before_after.png", dpi=150); plt.close()

# dashboard
fig = plt.figure(figsize=(16, 11))
fig.suptitle("Loan Applications - Cleaned Data Dashboard", fontsize=20, fontweight="bold")
gs = fig.add_gridspec(3, 3, hspace=.45, wspace=.3)

kpis = [("Applicants", f"{len(df):,}"),
        ("Default rate", f"{df.is_default.mean():.1%}"),
        ("Avg credit score", f"{df.credit_score.mean():.0f}")]
for i, (k, v) in enumerate(kpis):
    a = fig.add_subplot(gs[0, i]); a.axis("off")
    a.text(.5, .62, v, ha="center", fontsize=30, fontweight="bold", color="#1f4fd8")
    a.text(.5, .22, k, ha="center", fontsize=13, color="gray")

a = fig.add_subplot(gs[1, 0]); sns.histplot(df.age, bins=20, kde=True, ax=a, color="#4c72b0")
a.set_title("Age distribution")

a = fig.add_subplot(gs[1, 1]); sns.barplot(data=df, x="score_band", y="is_default", ax=a, errorbar=None, color="#dd8452")
a.set_title("Default rate by credit score band"); a.set_ylabel("Default rate"); a.set_xlabel("")

a = fig.add_subplot(gs[1, 2])
df.groupby("city")["loan_amount"].mean().sort_values().plot.barh(ax=a, color="#55a868")
a.set_title("Avg loan amount by city"); a.set_ylabel("")

a = fig.add_subplot(gs[2, 0]); sns.boxplot(data=df, x="default", y="loan_to_income", ax=a, hue="default", palette="Set2", legend=False)
a.set_title("Loan-to-income ratio vs default")

a = fig.add_subplot(gs[2, 1])
sns.heatmap(df[["age", "annual_income", "credit_score", "loan_amount", "loan_to_income", "is_default"]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=a, cbar=False)
a.set_title("Correlation matrix")

a = fig.add_subplot(gs[2, 2])
df.groupby("employment")["is_default"].mean().sort_values().plot.barh(ax=a, color="#c44e52")
a.set_title("Default rate by employment"); a.set_ylabel("")
plt.savefig("dashboard.png", dpi=150, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------
# 5. KEY FINDINGS (computed, not hand-typed)
# ---------------------------------------------------------------
by_band = df.groupby("score_band", observed=True)["is_default"].mean()
by_emp = df.groupby("employment")["is_default"].mean()
corr = df[["credit_score", "loan_to_income", "age", "annual_income"]].corrwith(df["is_default"])
findings = [
    f"Overall default rate is {df.is_default.mean():.1%}.",
    f"Default rate falls from {by_band.iloc[0]:.1%} (score <600) to {by_band.iloc[-1]:.1%} (score 750+).",
    f"Highest-risk employment group: {by_emp.idxmax()} ({by_emp.max():.1%}); lowest: {by_emp.idxmin()} ({by_emp.min():.1%}).",
    f"Strongest correlate of default: {corr.abs().idxmax()} (r = {corr[corr.abs().idxmax()]:.2f}).",
    f"Defaulters' median loan-to-income is {df[df.is_default==1].loan_to_income.median():.2f} vs {df[df.is_default==0].loan_to_income.median():.2f} for non-defaulters.",
]
print("\nKEY FINDINGS")
for f in findings:
    print("-", f)
with open("cleaning_log_and_findings.txt", "w") as fh:
    fh.write("CLEANING LOG\n" + "\n".join(log) + "\n\nKEY FINDINGS\n" + "\n".join("- " + f for f in findings))
