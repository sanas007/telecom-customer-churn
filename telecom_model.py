import os
import webbrowser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# === LOAD DATASET ===
csv_path = r"WA_Fn-UseC_-Telco-Customer-Churn.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found: {csv_path}")

df = pd.read_csv(csv_path)

# === CLEAN DATA ===
df.replace(" ", np.nan, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1}).astype(int)

# === REMOVE LEAKAGE COLUMNS ===
leakage_cols = ['customerID', 'Customer Churn Counter', 'Total counter', 'Tenure In years', 'Churn Count', 'Target']
for col in leakage_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# === ENCODE CATEGORICAL DATA ===
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
le = LabelEncoder()
encoded = df.copy()
for col in cat_cols:
    if encoded[col].nunique() == 2:
        encoded[col] = le.fit_transform(encoded[col])
    else:
        encoded = pd.get_dummies(encoded, columns=[col], prefix=col)

# === SPLIT DATA ===
X = encoded.drop(columns=['Churn'])
y = encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# === TRAIN MODEL ===
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# === MODEL PERFORMANCE ===
acc = accuracy_score(y_test, y_pred)
print("\n✅ MODEL EVALUATION RESULTS")
print("=" * 45)
print(f"Accuracy Score: {acc:.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

# === FEATURE IMPORTANCES ===
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
top5 = importances.head(5)
print("\n🔝 TOP 5 FEATURES INFLUENCING CHURN:")
for i, (feat, val) in enumerate(top5.items(), 1):
    print(f"{i:2d}. {feat:<40} {val:.4f}")

# === VISUALIZATION (4 KPI CHARTS) ===
plt.close('all')
fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=120)
fig.suptitle("TELECOM CUSTOMER CHURN", fontsize=15, fontweight='bold')
axes = axes.flatten()

# --- KPI 1: Overall Churn Rate ---
churn_pct = df['Churn'].value_counts(normalize=True).sort_index() * 100
bars1 = axes[0].bar(['Stayed', 'Left'], churn_pct.values, color=['#90caf9', '#f48fb1'], edgecolor='black')
axes[0].set_ylim(0, max(churn_pct.values) * 1.25)
axes[0].set_title("KPI 1 — What % of Customers Left vs Stayed", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Percentage (%)")
for bar, val in zip(bars1, churn_pct.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}%", ha='center', fontsize=10, fontweight='bold')

# --- KPI 2: Churn by Contract Type (Horizontal Bar Chart) ---
if 'Contract' in df.columns:
    contract_churn = df.groupby('Contract')['Churn'].mean().sort_values(ascending=True) * 100
    bars2 = axes[1].barh(contract_churn.index, contract_churn.values, color='#81c784', edgecolor='black')
    axes[1].set_xlim(0, 100)
    axes[1].set_title("KPI 2 — Which Contract Types Have Highest Churn", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Churn Rate (%)")
    for bar, val in zip(bars2, contract_churn.values):
        axes[1].text(val + 1.5, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va='center', fontsize=10, fontweight='bold')
else:
    axes[1].text(0.5, 0.5, "Contract column missing", ha='center')

# --- KPI 3: Churn by Internet Service Type ---
if 'InternetService' in df.columns:
    internet_churn = df.groupby('InternetService')['Churn'].mean().sort_values(ascending=False) * 100
    bars3 = axes[2].bar(internet_churn.index, internet_churn.values, color=['#ffcc80', '#ce93d8', '#80cbc4'], edgecolor='black')
    axes[2].set_ylim(0, 100)
    axes[2].set_title("KPI 3 — Churn by Internet Service Type", fontsize=12, fontweight='bold')
    for bar, val in zip(bars3, internet_churn.values):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}%", ha='center', fontsize=10, fontweight='bold')
else:
    axes[2].text(0.5, 0.5, "InternetService column missing", ha='center')

# --- KPI 4: Top 5 Features Influencing Churn ---
axes[3].set_title("KPI 4 — Which Features Influence Churn Most", fontsize=12, fontweight='bold')
custom_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
wedges, texts, autotexts = axes[3].pie(
    top5.values,
    labels=top5.index,
    autopct='%1.1f%%',
    startangle=120,
    colors=custom_colors,
    pctdistance=0.8,        # Move percentage text closer to edge
    labeldistance=1.1,      # Move labels slightly outward
)
# Improve font visibility
for t in texts:
    t.set_fontsize(8)
   
for at in autotexts:
    at.set_fontsize(6)
    at.set_fontweight('bold')

# === ADJUST SPACING ===
plt.subplots_adjust(
    left=0.059,
    right=0.95,
    top=0.868,
    bottom=0.076,
    hspace=0.486,
    wspace=0.354
)

plt.show()
