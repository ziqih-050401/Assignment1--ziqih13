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
