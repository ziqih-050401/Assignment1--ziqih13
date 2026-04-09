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

# Save to CSV
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
