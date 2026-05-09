# ============================================================
#  Spotify Popularity Prediction — CS439 Research Paper
#  Authors: Kartik Seth & Dhruv Patel (Rutgers University)
#  Python implementation, step by step

#    pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn spotipy kaggle
#
#  You also need:
#  1. A Kaggle API key (kaggle.json) to download the dataset
#  2. A Spotify Developer account for the Web API (client_id + client_secret)
#     → Create one at: https://developer.spotify.com/dashboard
# ============================================================


# ─────────────────────────────────────────────────────────────
# STEP 0 — Import all the libraries we'll need

# Each library has a specific job:
#   - pandas / numpy   → handle and crunch data
#   - matplotlib / seaborn → draw charts
#   - sklearn          → machine learning models and helpers
#   - xgboost          → the gradient boosting model (best performer in the paper)
#   - shap             → explain WHY the model made a prediction
#   - spotipy          → talk to the Spotify Web API

import os
import warnings
warnings.filterwarnings("ignore")       # keeps output clean

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             classification_report)

import xgboost as xgb
import shap

# Spotify Web API wrapper — comment out if you don't have credentials yet
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    SPOTIPY_AVAILABLE = True
except ImportError:
    SPOTIPY_AVAILABLE = False
    print("spotipy not installed — skipping Spotify API enrichment.")

print("✅  All libraries imported successfully!\n")


# =============================================================================
# STEP 1: Load the Dataset and Explore It
# =============================================================================
# The paper uses a Kaggle dataset of ~4,232 Spotify tracks from 2022.
# Download it from:
# https://www.kaggle.com/datasets/spotify-top-songs-2022
#
# Save it as: spotify_2022.csv in the same folder as these scripts.
# =============================================================================
# --- Load ---
df = pd.read_csv("dataset.csv")

# --- Basic shape ---
print("=== Dataset Shape ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

# --- Column names and types ---
print("=== Column Names & Data Types ===")
print(df.dtypes)
print()

# --- First few rows ---
print("=== First 5 Rows ===")
print(df.head())
print()

# --- Summary statistics for numeric columns ---
print("=== Summary Statistics ===")
print(df.describe())
print()

# --- Check for missing values ---
print("=== Missing Values Per Column ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
print(missing_df[missing_df["missing_count"] > 0])
print()

# --- Check the target variable: popularity ---
print("=== Target Variable: popularity ===")
print(f"Min:    {df['popularity'].min()}")
print(f"Max:    {df['popularity'].max()}")
print(f"Mean:   {df['popularity'].mean():.2f}")
print(f"Median: {df['popularity'].median()}")
print(f"Tracks with popularity = 0: {(df['popularity'] == 0).sum()}")


# ─────────────────────────────────────────────────────────────
# STEP 2 — (Optional) Enrich with Spotify Web API
# ─────────────────────────────────────────────────────────────
# The paper supplements the Kaggle data with artist-level popularity
# pulled directly from the Spotify API.  This gives each song a proxy
# for how famous the artist already is — the single strongest predictor
# found in the paper (SHAP rank #1).
#
# Replace the placeholder strings with your real credentials.
 
SPOTIPY_CLIENT_ID     = "YOUR_CLIENT_ID_HERE"
SPOTIPY_CLIENT_SECRET = "YOUR_CLIENT_SECRET_HERE"
 
def fetch_artist_popularity(artist_names: list) -> dict:
    """Return {artist_name: artist_popularity_score} using the Spotify API."""
    if not SPOTIPY_AVAILABLE:
        return {}
    if SPOTIPY_CLIENT_ID == "YOUR_CLIENT_ID_HERE":
        print("   Spotify credentials not set — skipping API enrichment.")
        return {}
 
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET
    ))
 
    artist_pop = {}
    for name in set(artist_names):
        try:
            results = sp.search(q=f"artist:{name}", type="artist", limit=1)
            items = results["artists"]["items"]
            if items:
                artist_pop[name] = items[0]["popularity"]
        except Exception:
            pass
    return artist_pop
 
# Only run if 'artist_popularity' column is missing
if "artist_popularity" not in df.columns:
    print("\nFetching artist popularity from Spotify API …")
    artist_map = fetch_artist_popularity(df["artist_name"].tolist())
    df["artist_popularity"] = df["artist_name"].map(artist_map).fillna(df["popularity"].median())
    print(f"   Fetched data for {len(artist_map)} unique artists.")
else:
    print("\n'artist_popularity' column already present — skipping API call.")
 
 
# ─────────────────────────────────────────────────────────────
# STEP 3 — Data Preprocessing
# ─────────────────────────────────────────────────────────────
# Raw data is messy. Before feeding it to a model we need to:
#   3a. Remove duplicate tracks
#   3b. Handle missing values
#   3c. Convert / normalize units
#   3d. Encode categorical columns as numbers
#   3e. Flag outlier tracks (popularity == 0)
 
print("\n" + "="*60)
print("STEP 3 — Data Preprocessing")
print("="*60)
 
# 3a ── Deduplication
# The paper removed exact duplicates and near-duplicates (same ISRC or
# >95% string similarity on title + artist).  We handle exact duplicates here.
before = len(df)
if "track_name" in df.columns and "artist_name" in df.columns:
    df = df.drop_duplicates(subset=["track_name", "artist_name"])
else:
    df = df.drop_duplicates()
print(f"3a. Deduplication: {before} → {len(df)} rows (removed {before - len(df)} duplicates)")
 
# 3b ── Missing value handling
# Drop columns with >20 % missing; fill remaining gaps with column medians.
threshold = 0.20
missing_frac = df.isnull().mean()
cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"3b. Dropped high-missingness columns: {cols_to_drop}")
 
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
print(f"3b. Imputed remaining numeric NaNs with column medians.")
 
# 3c ── Unit normalization
# Convert duration from milliseconds → minutes (easier to interpret).
# Normalize loudness (dB) from its raw range to [0, 1].
if "duration_ms" in df.columns:
    df["duration_min"] = df["duration_ms"] / 60_000
    df.drop(columns=["duration_ms"], inplace=True)
    print("3c. Converted duration_ms → duration_min")
 
if "loudness" in df.columns:
    scaler = MinMaxScaler()
    df["loudness"] = scaler.fit_transform(df[["loudness"]])
    print("3c. Normalized loudness to [0, 1]")
 
# 3d ── Encode categorical features
# Machine learning models need numbers, not text.
# Boolean columns (explicit) → 0 / 1
# release_type (album / single) → one-hot encoded columns
# Artist names with ≥10 appearances → label-encoded integer; rest → "Other"
 
if "explicit" in df.columns:
    df["explicit"] = df["explicit"].astype(int)
    print("3d. Encoded 'explicit' as 0/1")
 
if "release_type" in df.columns:
    df = pd.get_dummies(df, columns=["release_type"], prefix="release", drop_first=False)
    print("3d. One-hot encoded 'release_type'")
 
if "artist_name" in df.columns:
    artist_counts = df["artist_name"].value_counts()
    frequent_artists = artist_counts[artist_counts >= 10].index
    df["artist_name"] = df["artist_name"].apply(
        lambda x: x if x in frequent_artists else "Other"
    )
    le = LabelEncoder()
    df["artist_encoded"] = le.fit_transform(df["artist_name"])
    df.drop(columns=["artist_name"], inplace=True)
    print("3d. Label-encoded artist names (rare → 'Other')")
 
# Drop any remaining non-numeric columns (e.g. track_name, ISRC)
non_numeric = df.select_dtypes(exclude=[np.number, bool]).columns.tolist()
if non_numeric:
    df.drop(columns=non_numeric, inplace=True)
    print(f"3d. Dropped non-numeric columns: {non_numeric}")
 
# 3e ── Flag popularity == 0 (catalog / niche tracks)
# We keep them but note they may skew the distribution.
zero_pop = (df["popularity"] == 0).sum()
df["is_zero_popularity"] = (df["popularity"] == 0).astype(int)
print(f"3e. Flagged {zero_pop} tracks with popularity = 0")
 
print(f"\nPreprocessed dataset shape: {df.shape}")
 
 
# ─────────────────────────────────────────────────────────────
# STEP 4 — Exploratory Data Analysis (EDA)
# ─────────────────────────────────────────────────────────────
# Before modelling, explore the data visually.
# The paper found:
#   • Bimodal popularity distribution (cluster near 0 + bell curve ~55)
#   • Positive correlations: danceability (r=0.31), energy (r=0.27), artist_popularity (r=0.58)
#   • Negative correlation: acousticness (r=−0.22)
 
print("\n" + "="*60)
print("STEP 4 — Exploratory Data Analysis")
print("="*60)
 
os.makedirs("./plots", exist_ok=True)
 
# ── 4a: Popularity distribution
plt.figure(figsize=(8, 4))
plt.hist(df["popularity"], bins=40, color="#1DB954", edgecolor="black", alpha=0.85)
plt.title("Distribution of Song Popularity (0–100)", fontsize=14)
plt.xlabel("Popularity Score")
plt.ylabel("Number of Tracks")
plt.tight_layout()
plt.savefig("./plots/01_popularity_distribution.png", dpi=150)
plt.close()
print("4a. Saved popularity distribution chart → ./plots/01_popularity_distribution.png")
 
# ── 4b: Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
 
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, annot_kws={"size": 7})
plt.title("Pairwise Feature Correlations", fontsize=14)
plt.tight_layout()
plt.savefig("./plots/02_correlation_heatmap.png", dpi=150)
plt.close()
print("4b. Saved correlation heatmap → ./plots/02_correlation_heatmap.png")
 
# ── 4c: Print top correlations with popularity
pop_corr = corr["popularity"].drop("popularity").sort_values(ascending=False)
print("\nTop correlations with popularity:")
print(pop_corr.head(10).to_string())

# ─────────────────────────────────────────────────────────────
# STEP 5 — Prepare Features and Labels
# ─────────────────────────────────────────────────────────────
# Split the cleaned dataframe into:
#   X  → the input features the model will learn from
#   y  → the target variable (popularity score)
#
# We also create a binary label for the classification task:
#   popular = 1 if popularity >= 60, else 0   (paper's threshold)
 
TARGET = "popularity"
THRESHOLD = 60          # paper's threshold for "popular" vs "not popular"
 
# Drop the flag column we added in step 3e — it would leak the label
drop_cols = ["is_zero_popularity"]
feature_cols = [c for c in df.columns if c not in [TARGET] + drop_cols]
 
X = df[feature_cols]
y_reg = df[TARGET]                             # continuous, for regression
y_cls = (df[TARGET] >= THRESHOLD).astype(int)  # binary, for classification
 
print("\n" + "="*60)
print("STEP 5 — Feature / Label Split")
print("="*60)
print(f"Features used: {list(X.columns)}")
print(f"Total samples: {len(X)}")
print(f"Class balance (popular ≥ {THRESHOLD}): {y_cls.mean()*100:.1f}% popular")
 
 
# ─────────────────────────────────────────────────────────────
# STEP 6 — Train / Test Split
# ─────────────────────────────────────────────────────────────
# Hold out 20 % of data as a final test set.
# The paper used STRATIFIED sampling to ensure the test set has the
# same popularity distribution as training — important given the
# bimodal distribution we saw in EDA.
#
# The remaining 80 % is used for training (with 5-fold CV inside).
 
print("\n" + "="*60)
print("STEP 6 — Train / Test Split (80 / 20, stratified)")
print("="*60)
 
# For stratification on a continuous target we bin it into quartiles
strat_bins = pd.qcut(y_reg, q=4, labels=False, duplicates="drop")
 
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.20, random_state=42, stratify=strat_bins
)
_, _, y_train_cls, y_test_cls = train_test_split(
    X, y_cls, test_size=0.20, random_state=42, stratify=strat_bins
)
 
print(f"Training set : {X_train.shape[0]} samples")
print(f"Test set     : {X_test.shape[0]} samples")
 
 
# ─────────────────────────────────────────────────────────────
# STEP 7 — Train the Models
# ─────────────────────────────────────────────────────────────
# The paper trains three models and compares them:
#
#   Model 1: Ridge Regression (linear baseline)
#       Simple, interpretable, but limited by linearity.
#
#   Model 2: Random Forest (ensemble of decision trees)
#       Captures non-linear patterns; good balance of speed vs accuracy.
#
#   Model 3: XGBoost (gradient boosting)
#       Best performer in the paper. Builds trees sequentially,
#       each one correcting the errors of the previous.
#
# We train both the REGRESSION versions (predict a score 0–100)
# and the CLASSIFICATION versions (predict popular / not popular).
 
print("\n" + "="*60)
print("STEP 7 — Training Models")
print("="*60)
 
# ── Helper: pretty-print metrics ─────────────────────────────
def print_regression_metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:30s}  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}")
    return {"model": name, "RMSE": rmse, "MAE": mae, "R2": r2}
 
def print_classification_metrics(name, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"  {name:30s}  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
    return {"model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
 
reg_results  = []
cls_results  = []
 
# ── 7a: Ridge Regression (baseline) ──────────────────────────
print("\n7a. Ridge Regression (Baseline)")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train_reg)
ridge_pred = ridge.predict(X_test)
reg_results.append(print_regression_metrics("Ridge Regression", y_test_reg, ridge_pred))
 
# Classification equivalent: Logistic Regression
print("\n    Logistic Regression (classification baseline)")
logit = LogisticRegression(max_iter=500, random_state=42)
logit.fit(X_train, y_train_cls)
logit_pred = logit.predict(X_test)
cls_results.append(print_classification_metrics("Logistic Regression", y_test_cls, logit_pred))
 
# ── 7b: Random Forest ─────────────────────────────────────────
# 300 trees, max depth 15, min 5 samples per leaf
# (hyperparameters from the paper, tuned via 5-fold CV grid search)
print("\n7b. Random Forest")
rf_reg = RandomForestRegressor(
    n_estimators=300, max_depth=15, min_samples_leaf=5,
    max_features="sqrt", random_state=42, n_jobs=-1
)
rf_reg.fit(X_train, y_train_reg)
rf_pred = rf_reg.predict(X_test)
reg_results.append(print_regression_metrics("Random Forest", y_test_reg, rf_pred))
 
rf_cls = RandomForestClassifier(
    n_estimators=300, max_depth=15, min_samples_leaf=5,
    max_features="sqrt", random_state=42, n_jobs=-1
)
rf_cls.fit(X_train, y_train_cls)
rf_cls_pred = rf_cls.predict(X_test)
cls_results.append(print_classification_metrics("Random Forest", y_test_cls, rf_cls_pred))
 
# ── 7c: XGBoost ───────────────────────────────────────────────
# 500 estimators, lr=0.05, max depth 6, subsampling 0.8
# Early stopping prevents overfitting on a small validation hold-out.
print("\n7c. XGBoost (Gradient Boosting)")
 
# Split a small validation set out of training data for early stopping
X_tr, X_val, y_tr_r, y_val_r = train_test_split(
    X_train, y_train_reg, test_size=0.1, random_state=42
)
_, _, y_tr_c, y_val_c = train_test_split(
    X_train, y_train_cls, test_size=0.1, random_state=42
)
 
xgb_reg = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, random_state=42,
    early_stopping_rounds=20, eval_metric="rmse",
    verbosity=0
)
xgb_reg.fit(X_tr, y_tr_r, eval_set=[(X_val, y_val_r)], verbose=False)
xgb_pred = xgb_reg.predict(X_test)
reg_results.append(print_regression_metrics("XGBoost", y_test_reg, xgb_pred))
 
xgb_cls = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, random_state=42,
    early_stopping_rounds=20, eval_metric="logloss",
    verbosity=0, use_label_encoder=False
)
xgb_cls.fit(X_tr, y_tr_c, eval_set=[(X_val, y_val_c)], verbose=False)
xgb_cls_pred = xgb_cls.predict(X_test)
cls_results.append(print_classification_metrics("XGBoost", y_test_cls, xgb_cls_pred))
 
