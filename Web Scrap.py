import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

url = "https://bana290-assignment1.netlify.app/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

table = soup.find("table")

# Get ALL rows (headers + data are all in <tr> tags)
all_rows = table.find_all("tr")

# First row with content = headers
headers = []
for cell in all_rows[0].find_all(["th", "td"]):
    headers.append(cell.get_text(strip=True))

# If first row was empty (sometimes a spacer row), try the next
if not any(headers):
    headers = [cell.get_text(strip=True) for cell in all_rows[1].find_all(["th", "td"])]
    data_rows = all_rows[2:]
else:
    data_rows = all_rows[1:]

# Extract data rows
rows = []
for tr in data_rows:
    cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
    if cells and any(cells):  # skip empty rows
        rows.append(cells)

# Create DataFrame
df = pd.DataFrame(rows, columns=headers if headers else None)

# Strip segment + "profile" suffix from Firm (cell text is concatenated on the page)
if "Firm" in df.columns and "Segment" in df.columns:
    for seg in df["Segment"].unique():
        if pd.isna(seg):
            continue
        seg = str(seg).strip()
        df["Firm"] = df["Firm"].str.replace(seg + " profile", "", regex=False)
        df["Firm"] = df["Firm"].str.replace(seg + "profile", "", regex=False)
    df["Firm"] = df["Firm"].str.strip()

# Save to CSV (after Firm cleaning so the raw export matches downstream parsing)
df.to_csv("fintech_directory.csv", index=False)

print(f"Scraped {len(df)} firms with {len(df.columns)} columns")
print(df.head())

# Data Cleaning
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print(df.head())

# Data Analysis
print(df.describe())

# Data Visualization
plt.figure(figsize=(10, 6))

# Date Shape
print(df.shape)

# Data Type conversion
def parse_revenue(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    s = s.replace("usd", "").replace("$", "").replace(",", "").strip()

    if "million" in s:
        num = float(re.search(r"[\d.]+", s).group())
        return num * 1_000_000
    elif "mn" in s:
        num = float(re.search(r"[\d.]+", s).group())
        return num * 1_000_000
    elif "m" in s:
        num = float(re.search(r"[\d.]+", s).group())
        return num * 1_000_000
    else:
        return float(s)

def parse_customers(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(",", "")
    if s.upper().endswith("K"):
        return float(s[:-1]) * 1_000
    elif s.upper().endswith("M"):
        return float(s[:-1]) * 1_000_000
    return float(s)

def parse_pct(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace("+", "").replace("%", "")
    if s in ("--", "N/A", "n/a", ""):
        return np.nan
    return float(s)

def parse_rd(val, revenue):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    if s in ("--", "n/a", "unknown", ""):
        return np.nan

    # Handle "X% rev" percentage of annual revenue
    if "% rev" in s or "%rev" in s:
        pct = float(re.search(r"[\d.]+", s).group())
        if pd.notna(revenue):
            return (pct / 100) * revenue
        return np.nan

    s = s.replace("usd", "").replace("$", "").replace(",", "").strip()

    if "million" in s:
        return float(re.search(r"[\d.]+", s).group()) * 1_000_000
    elif "mn" in s:
        return float(re.search(r"[\d.]+", s).group()) * 1_000_000
    elif "m" in s:
        return float(re.search(r"[\d.]+", s).group()) * 1_000_000
    else:
        return float(s)

def parse_team(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip().replace(",", "")
    if "k" in s:
        return int(float(s.replace("k", "")) * 1000)
    return int(float(s))

def standardize_ai(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ai_yes:
        return 1
    elif s in ai_no:
        return 0
    elif s in ai_missing:
        return np.nan
    return np.nan

def standardize_cloud(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower().replace("-", " ")
    if "native" in s:
        return "Cloud-Native"
    elif "forward" in s:
        return "Cloud-Forward"
    elif "hybrid" in s:
        return "Hybrid"
    elif "legacy" in s:
        return "Legacy"
    return val

df["Annual Rev."] = df["Annual Rev."].apply(parse_revenue)

df["Rev Growth (YoY)"] = df["Rev Growth (YoY)"].apply(parse_pct)

df["R&D Spend"] = df.apply(
    lambda row: parse_rd(row["R&D Spend"], row["Annual Rev."]), axis=1
)

df["Team Size"] = df["Team Size"].apply(parse_team)

df["Digital Sales"] = df["Digital Sales"].apply(parse_pct)

df["Customer Accts"] = df["Customer Accts"].apply(parse_customers)

ai_yes = ["ai enabled", "yes", "adopted", "live", "production", "pilot"]
ai_no = ["no", "not yet", "legacy only", "manual only"]
ai_missing = ["--", "unknown", "n/a", "in review"]
df["AI Program"] = df["AI Program"].apply(standardize_ai)

df["Cloud Stack"] = df["Cloud Stack"].apply(standardize_cloud)

df["Compliance Tier"] = df["Compliance Tier"].str.extract(r"(\d)").astype(float)

print(f"\nBefore dropna: {df.shape[0]} rows")
print(f"NaN counts:\n{df.isna().sum()}\n")
df.dropna(inplace=True)
print(f"After dropna:  {df.shape[0]} rows")

df.to_csv("fintech_directory_cleaned.csv", index=False)
print(f"\nSaved cleaned data fintech_directory_cleaned.csv")
print(f"Final shape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")

df_model = pd.get_dummies(
    df,
    columns=["Segment", "Cloud Stack", "Fraud Exposure", "Funding Stage"],
    drop_first=True,
)

exclude = {"Firm", "HQ Region", "Rev Growth (YoY)", "AI Program"}
covariates = [c for c in df_model.columns if c not in exclude]

X = df_model[covariates].values.astype(float)
treatment = df_model["AI Program"].values.astype(int)
outcome = df_model["Rev Growth (YoY)"].values.astype(float)

X_ols = treatment.reshape(-1, 1)
ols = LinearRegression().fit(X_ols, outcome)
naive_coef = ols.coef_[0]
naive_intercept = ols.intercept_

# Manual significance test
n = len(outcome)
y_pred = ols.predict(X_ols)
residuals = outcome - y_pred
mse = np.sum(residuals ** 2) / (n - 2)
x_var = np.sum((treatment - treatment.mean()) ** 2)
if (n <= 2) or (x_var <= 0):
    se_coef = np.nan
    t_stat = np.nan
    p_value = np.nan
    r_squared = np.nan
    print(
        "WARNING: Cannot compute OLS inference (need treatment variation and n>2). "
        f"n={n}, x_var={x_var:.6g}"
    )
else:
    se_coef = np.sqrt(mse / x_var)
    t_stat = naive_coef / se_coef
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
    r_squared = 1 - np.sum(residuals ** 2) / np.sum((outcome - outcome.mean()) ** 2)

print("=" * 60)
print("STEP 1: NAIVE OLS — Rev Growth ~ AI Program")
print("=" * 60)
print(f"  Intercept:          {naive_intercept:.4f}")
print(f"  AI Coefficient:     {naive_coef:.4f}")
print(f"  Std Error:          {se_coef:.4f}")
print(f"  t-statistic:        {t_stat:.4f}")
print(f"  p-value:            {p_value:.4f}")
print(f"  R-squared:          {r_squared:.4f}")
direction = "increase" if naive_coef > 0 else "decrease"
print(f"    AI adoption is associated with a {naive_coef:.2f} pp {direction}")
print(f"    in YoY revenue growth (before controlling for confounders)")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ps_model = LogisticRegression(max_iter=1000, random_state=42)
ps_model.fit(X_scaled, treatment)
pscore = ps_model.predict_proba(X_scaled)[:, 1]

df_model["pscore"] = pscore
df_model["treatment"] = treatment

print("\n" + "=" * 60)
print("STEP 2a: PROPENSITY SCORE MODEL")
print("=" * 60)
print(f"  Model accuracy:            {ps_model.score(X_scaled, treatment):.2%}")
print(f"  P-score range (treated):   [{pscore[treatment==1].min():.3f}, {pscore[treatment==1].max():.3f}]")
print(f"  P-score range (control):   [{pscore[treatment==0].min():.3f}, {pscore[treatment==0].max():.3f}]")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax1 = axes[0]
ax1.hist(pscore[treatment == 1], bins=15, alpha=0.6, color="#2196F3",
         label="AI Adopted (T=1)", edgecolor="white")
ax1.hist(pscore[treatment == 0], bins=15, alpha=0.6, color="#FF5722",
         label="No AI (T=0)", edgecolor="white")
ax1.set_xlabel("Propensity Score", fontsize=12)
ax1.set_ylabel("Frequency", fontsize=12)
ax1.set_title("Common Support: Propensity Score Distributions",
              fontsize=13, fontweight="bold")
ax1.legend(fontsize=11)
ax1.grid(axis="y", alpha=0.3)

# KDE
ax2 = axes[1]
xs = np.linspace(0, 1, 200)
ps_t = pscore[treatment == 1]
ps_c = pscore[treatment == 0]
if (ps_t.size < 2) or (ps_c.size < 2):
    ax2.text(
        0.5,
        0.5,
        f"KDE skipped (need >=2 obs/group)\nT=1: {ps_t.size}, T=0: {ps_c.size}",
        ha="center",
        va="center",
        transform=ax2.transAxes,
        fontsize=11,
    )
else:
    kde_t = gaussian_kde(ps_t)
    kde_c = gaussian_kde(ps_c)
    ax2.fill_between(xs, kde_t(xs), alpha=0.4, color="#2196F3", label="AI Adopted (T=1)")
    ax2.fill_between(xs, kde_c(xs), alpha=0.4, color="#FF5722", label="No AI (T=0)")
    ax2.plot(xs, kde_t(xs), color="#1565C0", lw=2)
    ax2.plot(xs, kde_c(xs), color="#D84315", lw=2)
ax2.set_xlabel("Propensity Score", fontsize=12)
ax2.set_ylabel("Density", fontsize=12)
ax2.set_title("Common Support: Kernel Density Estimation",
              fontsize=13, fontweight="bold")
ax2.legend(fontsize=11)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("common_support.png", dpi=150, bbox_inches="tight")
plt.show()

def calc_smd(data, covariates, treatment_col):
    """Standardized Mean Difference for each covariate."""
    treated = data[data[treatment_col] == 1]
    control = data[data[treatment_col] == 0]
    smds = {}
    for cov in covariates:
        # Coerce to numeric to avoid object-dtype mean/std issues (e.g. "2002" as string)
        t_vals = pd.to_numeric(treated[cov], errors="coerce")
        c_vals = pd.to_numeric(control[cov], errors="coerce")
        mean_t = t_vals.mean()
        mean_c = c_vals.mean()
        std_t = t_vals.std()
        std_c = c_vals.std()
        pooled_std = np.sqrt((std_t ** 2 + std_c ** 2) / 2)
        smds[cov] = (mean_t - mean_c) / pooled_std if pooled_std > 0 else 0.0
    return smds


smd_before = calc_smd(df_model, covariates, "treatment")

print("\n" + "=" * 60)
print("STEP 2c: STANDARDIZED MEAN DIFFERENCES (BEFORE MATCHING)")
print("=" * 60)
print(f"  {'Covariate':<35} {'SMD':>8}")
print(f"  {'-'*35} {'-'*8}")
for cov, smd in sorted(smd_before.items(), key=lambda x: abs(x[1]), reverse=True):
    flag = " ***" if abs(smd) > 0.25 else " *" if abs(smd) > 0.1 else ""
    print(f"  {cov:<35} {smd:>8.4f}{flag}")

treated_idx = np.where(treatment == 1)[0]
control_idx = np.where(treatment == 0)[0]

# Fit nearest neighbors on control propensity scores
nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn.fit(pscore[control_idx].reshape(-1, 1))

# Match each treated unit to its nearest control
distances, indices = nn.kneighbors(pscore[treated_idx].reshape(-1, 1))
matched_control_idx = control_idx[indices.flatten()]

# Build matched dataset
treated_df = df_model.iloc[treated_idx].copy()
treated_df["matched_group"] = "treated"
control_df = df_model.iloc[matched_control_idx].copy()
control_df["matched_group"] = "control"
matched_df = pd.concat([treated_df, control_df], ignore_index=True)

# SMD after matching
smd_after = calc_smd(matched_df, covariates, "treatment")

print("\n" + "=" * 60)
print("STEP 3a: STANDARDIZED MEAN DIFFERENCES (AFTER MATCHING)")
print("=" * 60)
print(f"  {'Covariate':<35} {'Before':>8} {'After':>8} {'Improved?':>10}")
print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*10}")
for cov in sorted(smd_before.keys(), key=lambda x: abs(smd_before[x]), reverse=True):
    before = smd_before[cov]
    after = smd_after[cov]
    improved = "Yes" if abs(after) < abs(before) else "No"
    print(f"  {cov:<35} {before:>8.4f} {after:>8.4f} {improved:>10}")

att = outcome[treated_idx].mean() - outcome[matched_control_idx].mean()

np.random.seed(42)
n_boot = 1000
boot_atts = []
for _ in range(n_boot):
    boot_idx = np.random.choice(len(treated_idx), size=len(treated_idx), replace=True)
    boot_treated = outcome[treated_idx[boot_idx]]
    boot_control = outcome[matched_control_idx[boot_idx]]
    boot_atts.append(boot_treated.mean() - boot_control.mean())

att_se = np.std(boot_atts)
att_t = att / att_se if att_se > 0 else 0
att_p = 2 * (1 - stats.t.cdf(abs(att_t), df=len(treated_idx) - 1))
ci_lower = att - 1.96 * att_se
ci_upper = att + 1.96 * att_se

print("\n" + "=" * 60)
print("STEP 3b: PSM RESULTS — Average Treatment Effect on Treated (ATT)")
print("=" * 60)
print(f"  ATT (AI Adoption on Rev Growth):  {att:.4f} pp")
print(f"  Bootstrap SE:           {att_se:.4f}")
print(f"  t-statistic:            {att_t:.4f}")
print(f"  p-value:                {att_p:.4f}")
print(f"  95% CI:                 [{ci_lower:.4f}, {ci_upper:.4f}]")

print("\n" + "=" * 60)
print("COMPARISON: NAIVE OLS vs PSM")
print("=" * 60)
print(f"  Naive OLS coefficient:  {naive_coef:.4f} pp")
print(f"  PSM ATT estimate:       {att:.4f} pp")
print(f"  Difference:             {abs(naive_coef - att):.4f} pp")
sig_naive = "Significant" if p_value < 0.05 else "Not significant"
sig_psm = "Significant" if att_p < 0.05 else "Not significant"
print(f"  Naive OLS:              {sig_naive} (p={p_value:.4f})")
print(f"  PSM:                    {sig_psm} (p={att_p:.4f})")

fig, ax = plt.subplots(figsize=(10, 8))
cov_labels = sorted(smd_before.keys(), key=lambda x: abs(smd_before[x]))
y_pos = np.arange(len(cov_labels))
before_vals = [abs(smd_before[c]) for c in cov_labels]
after_vals = [abs(smd_after[c]) for c in cov_labels]

ax.scatter(before_vals, y_pos, color="#FF5722", s=80, zorder=3,
           label="Before Matching", marker="o")
ax.scatter(after_vals, y_pos, color="#2196F3", s=80, zorder=3,
           label="After Matching", marker="D")
for i in range(len(cov_labels)):
    ax.plot([before_vals[i], after_vals[i]], [y_pos[i], y_pos[i]],
            color="gray", alpha=0.4, lw=1)

ax.axvline(x=0.1, color="green", linestyle="--", alpha=0.7, label="SMD = 0.1 threshold")
ax.axvline(x=0.25, color="red", linestyle="--", alpha=0.7, label="SMD = 0.25 threshold")

short_labels = [
    c.replace("Segment_", "Seg: ")
     .replace("Cloud Stack_", "Cloud: ")
     .replace("Fraud Exposure_", "Fraud: ")
     .replace("Funding Stage_", "Fund: ")
    for c in cov_labels
]
ax.set_yticks(y_pos)
ax.set_yticklabels(short_labels, fontsize=9)
ax.set_xlabel("|Standardized Mean Difference|", fontsize=12)
ax.set_title("Love Plot: Covariate Balance Before vs After PSM",
             fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("love_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Saved: love_plot.png")

fig, ax = plt.subplots(figsize=(8, 5))
labels = ["Naive OLS", "PSM (ATT)"]
values = [naive_coef, att]
errors = [1.96 * se_coef, 1.96 * att_se]
colors = ["#FF5722", "#2196F3"]

bars = ax.bar(labels, values, yerr=errors, capsize=8,
              color=colors, edgecolor="white", width=0.5, alpha=0.85)
ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_ylabel("Effect of AI Adoption on Rev Growth (pp)", fontsize=12)
ax.set_title("Naive OLS vs Propensity Score Matching",
             fontsize=14, fontweight="bold")

for bar, val, pv in zip(bars, values, [p_value, att_p]):
    star = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "n.s."
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.2f} pp\n({star})", ha="center", va="bottom",
            fontsize=11, fontweight="bold")

ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("comparison_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: comparison_chart.png")

print("\n All analysis complete!")

"""
The naive OLS regression estimated that AI adoption is associated with a 4.49 percentage point increase in year-over-year revenue growth (p < 0.001, R² = 0.27). However, after applying Propensity Score Matching to control for observable confounders, the Average Treatment Effect on the Treated (ATT) dropped to 2.30 percentage points (p = 0.004, 95% CI: [0.81, 3.79]). While the effect remains statistically significant, its magnitude was roughly cut in half.
This shift strongly suggests that selection bias inflates naive estimates of AI's impact. Firms that adopt AI are not randomly selected, they tend to be larger, better-funded, concentrated in high-growth segments like Payments, and invest more heavily in R&D. The naive OLS conflates these pre-existing advantages with the causal effect of AI itself. Without adjustment, the superior performance of AI may largely reflect the characteristics of firms that adopt it, not the technology's standalone contribution.
Regarding the validity of PSM assumptions, the common support condition was reasonably satisfied. The propensity score distributions for treated (0.22–0.98) and control (0.01–0.84) groups showed substantial overlap, meaning suitable matches existed across most of the score range. The balancing property, however, was only partially achieved. While several covariates improved after matching others, particularly continuous variables like Team Size, R&D Spend, and Founded year,  showed increased imbalance post-match. This is a recognized limitation of 1:1 nearest-neighbor matching on small samples, and suggests that results should be interpreted with caution. Caliper-based matching or inverse probability weighting could improve balance in future analyses.
"""