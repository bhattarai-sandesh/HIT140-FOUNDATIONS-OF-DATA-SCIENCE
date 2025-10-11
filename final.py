# =====================================
# BAT vs RAT ANALYSIS PROJECT (HIT140)
# =====================================
# Author: Saurav Baral (and team)
# Unit: HIT140 - Foundations of Data Science
# Assessment 3 - Group Project Report
# -------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# ================================
# STEP 1: LOAD DATA
# ================================
print("=== STEP 1: LOAD & INSPECT DATA ===")

df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")

print(f"Dataset1 shape: {df1.shape}")
print(f"Dataset2 shape: {df2.shape}")
print("\nMissing values (Dataset1):\n", df1.isna().sum())
print("\nMissing values (Dataset2):\n", df2.isna().sum())

# ================================
# STEP 2: CLEANING & PREPROCESSING
# ================================
print("\n=== STEP 2: CLEAN & PREPROCESS ===")

# Convert datetime columns
datetime_cols_df1 = ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]
datetime_cols_df2 = ["time"]

for col in datetime_cols_df1:
    df1[col] = pd.to_datetime(df1[col], dayfirst=True, errors='coerce')

for col in datetime_cols_df2:
    df2[col] = pd.to_datetime(df2[col], dayfirst=True, errors='coerce')

# Fill missing 'habit' values with mode
if "habit" in df1.columns:
    mode_value = df1["habit"].mode()[0]
    df1["habit"].fillna(mode_value, inplace=True)
    print(f"Filled missing 'habit' values with mode: {mode_value}")

# Replace negatives with NaN and fill with median
for col in df1.select_dtypes(include=np.number).columns:
    df1[col] = df1[col].apply(lambda x: np.nan if x < 0 else x)
df1.fillna(df1.median(numeric_only=True), inplace=True)

for col in df2.select_dtypes(include=np.number).columns:
    df2[col] = df2[col].apply(lambda x: np.nan if x < 0 else x)
df2.fillna(df2.median(numeric_only=True), inplace=True)

# Save clean copies
df1.to_csv("dataset1_clean.csv", index=False)
df2.to_csv("dataset2_clean.csv", index=False)
print("✅ Cleaning complete and cleaned files saved.\n")

# ================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ================================
print("=== STEP 3: EXPLORATORY DATA ANALYSIS ===")

sns.set(style="whitegrid", palette="viridis")
os.makedirs("plots", exist_ok=True)

def save_and_show(fig_name):
    plt.tight_layout()
    plt.savefig(f"plots/{fig_name}.png", dpi=300)
    plt.show()

# --- Risk-taking distribution ---
plt.figure(figsize=(6, 4))
sns.countplot(data=df1, x='risk')
plt.title("Distribution of Risk-taking Behaviour (0 = Avoid, 1 = Take Risk)")
plt.xlabel("Risk Behaviour")
plt.ylabel("Count")
save_and_show("risk_distribution")

# --- Reward vs Risk ---
plt.figure(figsize=(6, 4))
sns.countplot(data=df1, x='reward', hue='risk')
plt.title("Reward vs Risk Behaviour")
plt.xlabel("Reward (0=No, 1=Yes)")
plt.ylabel("Count")
save_and_show("reward_vs_risk")

# --- Risk-taking by Season ---
plt.figure(figsize=(7, 5))
sns.countplot(data=df1, x='season', hue='risk')
plt.title("Risk-taking Behaviour by Season")
plt.xlabel("Season")
plt.ylabel("Count")
save_and_show("risk_by_season")

# --- Rat Presence vs Bat Landings ---
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df2, x='rat_minutes', y='bat_landing_number', hue='hours_after_sunset')
plt.title("Rat Presence vs Bat Landings")
plt.xlabel("Rat Minutes (Presence Duration)")
plt.ylabel("Number of Bat Landings")
save_and_show("rat_vs_bat")

# --- Correlation Heatmap ---
plt.figure(figsize=(6, 5))
sns.heatmap(df2.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap (Dataset 2)")
save_and_show("correlation_heatmap")

print("✅ EDA plots displayed and saved in 'plots/' folder.\n")

# ================================
# STEP 4: STATISTICAL TESTING & MODELLING
# ================================
print("=== STEP 4: STATISTICAL TESTING & MODELLING ===")

# --- Investigation A: Rats as predators or competitors ---
risk_takers = df1[df1['risk'] == 1]['seconds_after_rat_arrival']
risk_avoiders = df1[df1['risk'] == 0]['seconds_after_rat_arrival']
t_stat, p_val = ttest_ind(risk_takers, risk_avoiders, equal_var=False, nan_policy='omit')
print(f"\nT-Test (Risk-takers vs Avoiders): t = {t_stat:.3f}, p = {p_val:.5f}")

if p_val < 0.05:
    print("✅ Significant: bats behave differently when rats are present (fear effect).")
else:
    print("⚪ Not significant: bats likely see rats as competitors, not predators.")

# --- Investigation B: Seasonal behaviour ---
season_risk_table = pd.crosstab(df1['season'], df1['risk'])
chi2, p_chi, dof, expected = chi2_contingency(season_risk_table)
print(f"\nChi-Square (Season vs Risk): χ² = {chi2:.3f}, p = {p_chi:.5f}")

if p_chi < 0.05:
    print("✅ Significant: bat risk behaviour changes by season.")
else:
    print("⚪ No significant seasonal effect detected.")

# --- Rat Activity vs Bat Landings ---
corr, p_corr = pearsonr(df2['rat_minutes'], df2['bat_landing_number'])
print(f"\nPearson Correlation (Rat Minutes vs Bat Landings): r = {corr:.3f}, p = {p_corr:.5f}")

if p_corr < 0.05:
    print("✅ Significant correlation: rat activity affects bat landings.")
else:
    print("⚪ No strong correlation detected between rat and bat activity.")

# --- Logistic Regression Model ---
print("\n=== Logistic Regression Model: Predicting Risk Behaviour ===")

features = ['bat_landing_to_food', 'seconds_after_rat_arrival', 'reward', 'hours_after_sunset']
X = df1[features]
y = df1['risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Feature Importance ---
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nFeature Importance:")
print(coef_df)

plt.figure(figsize=(7, 4))
sns.barplot(data=coef_df, x='Feature', y='Coefficient', palette='viridis', legend=False)
plt.title("Feature Importance (Logistic Regression)")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=300)
plt.show()

print("\n✅ Statistical testing & modelling completed successfully.")
print("All plots displayed and saved — ready for report writing! 🦇🐀📊")