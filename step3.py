# =====================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# =====================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned datasets
df1 = pd.read_csv("dataset1_clean.csv")
df2 = pd.read_csv("dataset2_clean.csv")

# Set style
sns.set(style="whitegrid", palette="viridis")

# ---------- 1️⃣ Quick overall checks ----------
print("Dataset 1 (Bat Landings) shape:", df1.shape)
print("Dataset 2 (Rat Activity) shape:", df2.shape)

print("\nUnique seasons in Dataset 1:", df1['season'].unique())
print("Unique months in Dataset 1:", df1['month'].unique())

# ---------- 2️⃣ Distribution of risk-taking behaviour ----------
plt.figure(figsize=(6, 4))
sns.countplot(data=df1, x='risk')
plt.title("Distribution of Risk-taking Behaviour (0 = Avoid, 1 = Take Risk)")
plt.xlabel("Risk Behaviour")
plt.ylabel("Count")
plt.show()

# ---------- 3️⃣ Reward vs Risk relationship ----------
plt.figure(figsize=(6, 4))
sns.countplot(data=df1, x='reward', hue='risk')
plt.title("Reward vs Risk Behaviour")
plt.xlabel("Reward (0=No, 1=Yes)")
plt.ylabel("Count")
plt.legend(title="Risk")
plt.show()

# ---------- 4️⃣ Risk-taking by Season ----------
plt.figure(figsize=(7, 5))
sns.countplot(data=df1, x='season', hue='risk')
plt.title("Risk-taking Behaviour by Season")
plt.xlabel("Season")
plt.ylabel("Count")
plt.legend(title="Risk Type")
plt.show()

# ---------- 5️⃣ Average Rat Presence vs Bat Landings ----------
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df2, x='rat_minutes', y='bat_landing_number', hue='hours_after_sunset')
plt.title("Rat Presence vs Bat Landings")
plt.xlabel("Rat Minutes (Presence Duration)")
plt.ylabel("Number of Bat Landings")
plt.show()

# ---------- 6️⃣ Relationship between Rat Arrivals and Food Availability ----------
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df2, x='rat_arrival_number', y='food_availability')
plt.title("Rat Arrivals vs Remaining Food")
plt.xlabel("Number of Rat Arrivals")
plt.ylabel("Food Availability")
plt.show()

# ---------- 7️⃣ Correlation Heatmap (Dataset 2) ----------
plt.figure(figsize=(6, 5))
sns.heatmap(df2.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap (Dataset 2)")
plt.show()
