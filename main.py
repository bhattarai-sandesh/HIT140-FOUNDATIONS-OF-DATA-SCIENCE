# =====================================
# STEP 1: LOAD AND INSPECT THE DATASETS
# =====================================

# 1️⃣ Import essential libraries
import pandas as pd
import numpy as np

# 2️⃣ Load both CSV files
dataset1_path = "dataset1.csv"
dataset2_path = "dataset2.csv"

# Read the files
df1 = pd.read_csv(dataset1_path)
df2 = pd.read_csv(dataset2_path)

# 3️⃣ Basic information
print("===== DATASET 1: Bat Landings =====")
print(f"Shape: {df1.shape}")
print("\nColumns:\n", df1.columns.tolist())
print("\nFirst 5 rows:")
print(df1.head())

print("\nMissing values summary:")
print(df1.isna().sum())

print("\n===== DATASET 2: Rat Activity =====")
print(f"Shape: {df2.shape}")
print("\nColumns:\n", df2.columns.tolist())
print("\nFirst 5 rows:")
print(df2.head())

print("\nMissing values summary:")
print(df2.isna().sum())

# 4️⃣ Quick numerical summary (helps spot weird values)
print("\n--- DESCRIPTIVE SUMMARY: DATASET 1 ---")
print(df1.describe(include='all'))

print("\n--- DESCRIPTIVE SUMMARY: DATASET 2 ---")
print(df2.describe(include='all'))

# 5️⃣ Optional: Check datatypes to prep for cleaning
print("\nData types in Dataset 1:")
print(df1.dtypes)

print("\nData types in Dataset 2:")
print(df2.dtypes)

