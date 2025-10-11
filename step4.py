# =====================================
# STEP 4: STATISTICAL TESTING & MODELLING
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load cleaned datasets
df1 = pd.read_csv("dataset1_clean.csv")
df2 = pd.read_csv("dataset2_clean.csv")

# ---------- 1️⃣ Investigation A: Risk-taking vs Rat Arrival ----------
print("=== Investigation A: Do bats perceive rats as predators? ===")

# Compare risk-taking with seconds_after_rat_arrival (time since rats appeared)
risk_takers = df1[df1['risk'] == 1]['seconds_after_rat_arrival']
risk_avoiders = df1[df1['risk'] == 0]['seconds_after_rat_arrival']

t_stat, p_val = ttest_ind(risk_takers, risk_avoiders, equal_var=False, nan_policy='omit')
print(f"\nT-Test between risk-taking and avoiding bats:")
print(f"T-statistic: {t_stat:.3f}, p-value: {p_val:.5f}")

if p_val < 0.05:
    print("✅ Statistically significant difference → suggests rats influence bat risk behaviour.")
else:
    print("⚪ No significant difference → bats might not perceive rats as a major threat.")

# ---------- 2️⃣ Investigation B: Season vs Risk-taking ----------
print("\n=== Investigation B: Seasonal differences in bat behaviour ===")

# Cross-tab of season vs risk
season_risk_table = pd.crosstab(df1['season'], df1['risk'])
chi2, p_val_chi, dof, expected = chi2_contingency(season_risk_table)

print("\nChi-square Test (Season vs Risk):")
print(f"Chi2: {chi2:.3f}, p-value: {p_val_chi:.5f}")

if p_val_chi < 0.05:
    print("✅ Risk-taking behaviour significantly changes by season.")
else:
    print("⚪ No strong seasonal effect detected.")

# ---------- 3️⃣ Investigation B extended: Correlation in dataset2 ----------
print("\n=== Dataset 2 Relationship Tests ===")

# Correlation between rat_minutes and bat_landing_number
corr, p_corr = pearsonr(df2['rat_minutes'], df2['bat_landing_number'])
print(f"Correlation between Rat Minutes and Bat Landings: {corr:.3f}, p = {p_corr:.5f}")

if p_corr < 0.05:
    print("✅ Significant correlation between rat activity and bat landings.")
else:
    print("⚪ No strong correlation between rat activity and bat landings.")

# ---------- 4️⃣ Logistic Regression: Predicting Risk ----------
print("\n=== Logistic Regression Model: Predicting Bat Risk Behaviour ===")

# Prepare features
features = ['bat_landing_to_food', 'seconds_after_rat_arrival', 'reward', 'hours_after_sunset']
X = df1[features]
y = df1['risk']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train logistic model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------- 5️⃣ Feature Importance ----------
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nFeature Importance:")
print(coef_df)

# Visualise feature importance
plt.figure(figsize=(7, 4))
sns.barplot(data=coef_df, x='Feature', y='Coefficient', palette='viridis')
plt.title("Feature Importance (Logistic Regression)")
plt.tight_layout()
plt.show()
