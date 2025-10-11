# =====================================
# STEP 2: DATA CLEANING & PREPROCESSING
# =====================================

import pandas as pd
import numpy as np

# Load again (if not already)
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")

# ---------- 1️⃣ Convert datetime columns ----------
datetime_cols_df1 = ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]
datetime_cols_df2 = ["time"]

for col in datetime_cols_df1:
    df1[col] = pd.to_datetime(df1[col], errors='coerce')

for col in datetime_cols_df2:
    df2[col] = pd.to_datetime(df2[col], errors='coerce')

print("✅ Datetime columns converted successfully.")

# ---------- 2️⃣ Handle missing values ----------
print("\nMissing values BEFORE cleaning (Dataset1):")
print(df1.isna().sum())

# Fill 'habit' missing values with the most common (mode)
if "habit" in df1.columns:
    mode_value = df1["habit"].mode()[0]
    df1["habit"].fillna(mode_value, inplace=True)
    print(f"\nFilled missing 'habit' values with mode: {mode_value}")

# Check again
print("\nMissing values AFTER cleaning (Dataset1):")
print(df1.isna().sum())

# ---------- 3️⃣ Clean any invalid numeric values ----------
# Replace negative or unrealistic values with NaN if they exist
num_cols_df1 = df1.select_dtypes(include=np.number).columns
for col in num_cols_df1:
    df1[col] = df1[col].apply(lambda x: np.nan if x < 0 else x)

num_cols_df2 = df2.select_dtypes(include=np.number).columns
for col in num_cols_df2:
    df2[col] = df2[col].apply(lambda x: np.nan if x < 0 else x)

# Fill any remaining missing numeric values with median
df1.fillna(df1.median(numeric_only=True), inplace=True)
df2.fillna(df2.median(numeric_only=True), inplace=True)

print("\n✅ Numeric columns cleaned and missing values handled.")

# ---------- 4️⃣ Verify cleaning ----------
print("\nDataset 1 info after cleaning:")
print(df1.info())
print("\nDataset 2 info after cleaning:")
print(df2.info())

# ---------- 5️⃣ Save cleaned versions (optional) ----------
df1.to_csv("dataset1_clean.csv", index=False)
df2.to_csv("dataset2_clean.csv", index=False)
print("\n💾 Cleaned datasets saved as 'dataset1_clean.csv' and 'dataset2_clean.csv'")
