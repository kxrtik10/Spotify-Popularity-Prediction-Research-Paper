"""
Predicting Song Popularity on Spotify
======================================
Authors : Kartik Seth (ks2221) | Dhruv Patel (dp1489)
Course  : CS439 — Rutgers University
Dataset : Kaggle "Spotify Top Songs 2022" by Amitansh Joshi
"""

# Imports 
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score)
import shap

# Paths
SCRIPT_DIR   = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "dataset.csv"
OUT_DIR      = SCRIPT_DIR / "outputs"
PLOT_DIR     = OUT_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

POPULARITY_THRESHOLD = 60
RANDOM_STATE         = 42

# STEP 1 — Load Dataset

print("=" * 60)
print("STEP 1 — Load Dataset")
print("=" * 60)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"\n[ERROR] dataset.csv not found at: {DATASET_PATH}\n"
        "Place dataset.csv in the same folder as this script and try again."
    )

df = pd.read_csv(DATASET_PATH)

print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
print(f"\nTarget variable (popularity) summary:\n{df['popularity'].describe().round(2)}")
print(f"Tracks with popularity = 0: {(df['popularity'] == 0).sum()}")

# STEP 2 — Summary Statistics 

print("\n" + "=" * 60)
print("STEP 2 — Summary Statistics")
print("=" * 60)

print(df.describe().round(2).to_string())

print(f"\nMissing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0].to_string() if missing.any() else "  None")
print("\n" + "=" * 60)
print("STEP 3 — Preprocessing")
print("=" * 60)

before = len(df)
subset_cols = ["track_name", "artist_name"] if {"track_name", "artist_name"}.issubset(df.columns) else None
df = df.drop_duplicates(subset=subset_cols)
print(f"3a. Deduplication: {before} → {len(df)} rows ({before - len(df)} removed)")
missing_frac = df.isnull().mean()
cols_to_drop = missing_frac[missing_frac > 0.20].index.tolist()
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"3b. Dropped high-missingness columns: {cols_to_drop}")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
print(f"3b. Imputed remaining numeric NaNs with column medians.")

if "duration_ms" in df.columns:
    df["duration_min"] = df["duration_ms"] / 60_000
    df = df.drop(columns=["duration_ms"])
    print("3c. Converted duration_ms → duration_min")

if "loudness" in df.columns:
    scaler = MinMaxScaler()
    df["loudness"] = scaler.fit_transform(df[["loudness"]])
    print("3c. Normalized loudness to [0, 1]")

if "explicit" in df.columns:
    df["explicit"] = df["explicit"].astype(int)
    print("3d. Encoded 'explicit' as 0/1")

if "release_type" in df.columns:
    if df["release_type"].nunique() <= 2:
        df["release_type"] = (df["release_type"].str.lower() == "single").astype(int)
        print("3d. Encoded 'release_type' as binary (single=1, album=0)")
    else:
        df = pd.get_dummies(df, columns=["release_type"], prefix="release", drop_first=False)
        print("3d. One-hot encoded 'release_type'")

if "artist_name" in df.columns:
    artist_counts = df["artist_name"].value_counts()
    frequent      = artist_counts[artist_counts >= 10].index
    df["artist_name"]    = df["artist_name"].apply(lambda x: x if x in frequent else "Other")
    df["artist_encoded"] = LabelEncoder().fit_transform(df["artist_name"])
    df = df.drop(columns=["artist_name"])
    print("3d. Label-encoded artist names (rare artists → 'Other')")
non_numeric = df.select_dtypes(exclude=[np.number, bool]).columns.tolist()
if non_numeric:
    df = df.drop(columns=non_numeric)
    print(f"3d. Dropped remaining non-numeric columns: {non_numeric}")
zero_pop_count = (df["popularity"] == 0).sum()
df["is_zero_pop"] = (df["popularity"] == 0).astype(int)
print(f"3e. Flagged {zero_pop_count} tracks with popularity = 0")

print(f"\nPreprocessed shape: {df.shape}")

# STEP 4 — Exploratory Data Analysis

print("\n" + "=" * 60)
print("STEP 4 — Exploratory Data Analysis")
print("=" * 60)

plt.figure(figsize=(8, 4))
plt.hist(df["popularity"], bins=40, color="#1DB954", edgecolor="black", alpha=0.85)
plt.axvline(df["popularity"].mean(), color="red", linestyle="--",
            label=f"Mean = {df['popularity'].mean():.1f}")
plt.title("Distribution of Song Popularity (0–100)", fontsize=14)
plt.xlabel("Popularity Score")
plt.ylabel("Number of Tracks")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "01_popularity_distribution.png", dpi=150)
plt.close()
print("4a. Saved → plots/01_popularity_distribution.png")
corr_cols   = [c for c in df.columns if c != "is_zero_pop"]
corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5, annot_kws={"size": 7})
plt.title("Pairwise Feature Correlations", fontsize=14)
plt.tight_layout()
plt.savefig(PLOT_DIR / "02_correlation_heatmap.png", dpi=150)
plt.close()
print("4b. Saved → plots/02_correlation_heatmap.png")

pop_corr = corr_matrix["popularity"].drop("popularity").sort_values(ascending=False)
print(f"\nPearson correlations with popularity (all features):\n{pop_corr.round(3).to_string()}")

# STEP 5 — Feature / Label Split

print("\n" + "=" * 60)
print("STEP 5 — Feature / Label Split")
print("=" * 60)

TARGET       = "popularity"
EXCLUDE      = [TARGET, "is_zero_pop"]
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE]

X     = df[FEATURE_COLS].values
y_reg = df[TARGET].values
y_cls = (df[TARGET] >= POPULARITY_THRESHOLD).astype(int).values

print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Samples : {len(X)}")
print(f"Class balance — popular (≥{POPULARITY_THRESHOLD}): {y_cls.mean()*100:.1f}%")

# STEP 6 — Train / Test Split (80/20, stratified on popularity quartiles)

print("\n" + "=" * 60)
print("STEP 6 — Train / Test Split")
print("=" * 60)

strat_bins = pd.qcut(y_reg, q=4, labels=False, duplicates="drop")

X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.20, random_state=RANDOM_STATE, stratify=strat_bins
)
_, _, y_train_cls, y_test_cls = train_test_split(
    X, y_cls, test_size=0.20, random_state=RANDOM_STATE, stratify=strat_bins
)

print(f"Training : {X_train.shape[0]} samples")
print(f"Test     : {X_test.shape[0]} samples")

# STEP 7 — Train Models

print("\n" + "=" * 60)
print("STEP 7 — Training Models")
print("=" * 60)

def reg_metrics(name, y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    print(f"  {name:40s}  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")
    return {"Model": name, "RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}

def cls_metrics(name, y_true, y_pred):
    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    print(f"  {name:40s}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    return {"Model": name, "Accuracy": round(acc, 4), "Precision": round(prec, 4),
            "Recall": round(rec, 4), "F1": round(f1, 4)}

reg_results = []
cls_results = []
print("\n7a. Linear Baselines")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train_reg)
reg_results.append(reg_metrics("Ridge Regression", y_test_reg, ridge.predict(X_test)))

logit = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
logit.fit(X_train, y_train_cls)
cls_results.append(cls_metrics("Logistic Regression", y_test_cls, logit.predict(X_test)))
print("\n7b. Random Forest")
rf_reg = RandomForestRegressor(
    n_estimators=300, max_depth=15, min_samples_leaf=5,
    max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1
)
rf_reg.fit(X_train, y_train_reg)
reg_results.append(reg_metrics("Random Forest", y_test_reg, rf_reg.predict(X_test)))

rf_cls = RandomForestClassifier(
    n_estimators=300, max_depth=15, min_samples_leaf=5,
    max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1
)
rf_cls.fit(X_train, y_train_cls)
cls_results.append(cls_metrics("Random Forest", y_test_cls, rf_cls.predict(X_test)))
print("\n7c. Gradient Boosting")

gb_reg = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, random_state=RANDOM_STATE
)
gb_reg.fit(X_train, y_train_reg)
gb_pred = gb_reg.predict(X_test)
reg_results.append(reg_metrics("Gradient Boosting", y_test_reg, gb_pred))

gb_cls = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, random_state=RANDOM_STATE
)
gb_cls.fit(X_train, y_train_cls)
cls_results.append(cls_metrics("Gradient Boosting", y_test_cls, gb_cls.predict(X_test)))
