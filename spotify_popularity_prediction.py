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