import argparse
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings("ignore")

# Features defined according to the new dataset structure
ROAD_FEATURES = [
    "length", "lanes", "building_density", "vegetation_score",
    "maxspeed", "width"
]

TRAFFIC_FEATURES = [
    "vehicle_count", "avg_speed_kmph",
    "n_car", "n_motorcycle", "n_bus",
    "n_truck", "n_van", "n_bicycle", "n_auto"
]

EMISSION_FEATURES = [
    "co2_per_km_g", "base_cost_feature",
    "NOx_BGC", "NOx_IND", "NOx_OTH", "NOx_RES", "NOx_TRA"
]

ENV_FEATURES = [
    "pm2_5_ugm3", "pm10_ugm3", "no2_ugm3", "o3_ugm3",
    "so2_ugm3", "co_ugm3", "AQI", "openweather_aqi_1to5",
    "temperature_k", "humidity_pct", "wind_speed_mps", "wind_dir_deg"
]

TEMPORAL_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]

CATEGORICAL_COLS = [
    "highway", "weather_description", "junction_enc",
    "oneway_enc", "reversed_enc", "bridge_enc", "landuse_enc"
]

TARGET = "carbon_cost"


def load_and_preprocess(data_path: str):
    print(f"[INFO] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"[INFO] Total rows loaded: {len(df):,}")

    # Create cyclic temporal features
    if "hour_of_day" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    if "day_of_week" in df.columns:
        df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Ensure all numerical features exist and fill missing values
    all_num = ROAD_FEATURES + TRAFFIC_FEATURES + EMISSION_FEATURES + ENV_FEATURES + TEMPORAL_FEATURES
    for col in all_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median() if df[col].dtype != object else 0)
        else:
            df[col] = 0.0

    # Categorical encodings (handle missing and stringify)
    df["junction_enc"] = df.get("junction", pd.Series(["none"] * len(df))).fillna("none").astype(str)
    df["oneway_enc"] = df.get("oneway", pd.Series([False] * len(df))).astype(str)
    df["reversed_enc"] = df.get("reversed", pd.Series([False] * len(df))).astype(str)
    df["bridge_enc"] = df.get("bridge", pd.Series(["no"] * len(df))).fillna("no").astype(str)
    df["landuse_enc"] = df.get("landuse", pd.Series(["unknown"] * len(df))).fillna("unknown").astype(str)
    df["weather_description"] = df.get("weather_description", pd.Series(["clear sky"] * len(df))).fillna("clear sky").astype(str)
    df["highway"] = df.get("highway", pd.Series(["residential"] * len(df))).fillna("residential").astype(str)

    # Prepare target
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    
    # We predict the log of the carbon cost to handle outliers/variance
    df[TARGET] = np.log1p(df[TARGET])

    features = all_num + CATEGORICAL_COLS
    return df[features], df[TARGET], features


def metrics_str(preds_log, trues_log):
    p = np.expm1(preds_log)
    t = np.expm1(trues_log)
    rmse = math.sqrt(mean_squared_error(t, p))
    mae  = mean_absolute_error(t, p)
    r2   = r2_score(t, p)
    mask = t > 1.0
    mape = np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100 if mask.sum() > 0 else 0
    return rmse, mae, r2, mape


def plot_pred_vs_actual(preds_log, trues_log, save_path="pred_vs_actual_catboost.png"):
    p = np.expm1(preds_log)
    t = np.expm1(trues_log)
    cap = np.percentile(t, 99)
    mask = t < cap
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(t[mask], p[mask], alpha=0.25, s=6, color="#4C72B0")
    lo, hi = t[mask].min(), t[mask].max()
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect Fit")
    ax.set_xlabel("Actual Carbon Cost (g CO2)")
    ax.set_ylabel("Predicted Carbon Cost")
    ax.set_title("Predicted vs Actual (CatBoost)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    residuals = p[mask] - t[mask]
    ax.hist(residuals, bins=80, color="#55A868", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="red", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Residual (Predicted - Actual)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved prediction plot -> {save_path}")


def main():
    parser = argparse.ArgumentParser(description="CatBoost Carbon Cost Predictor")
    parser.add_argument("--data", default="warsaw_roads_nox.csv", help="Path to the dataset CSV")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of CatBoost iterations")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--depth", type=int, default=6, help="Tree depth")
    parser.add_argument("--out-dir", default=".", help="Directory to save the model and plots")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading and preprocessing data...")
    X, y, feature_names = load_and_preprocess(args.data)
    print(f"      Features: {len(feature_names)}")
    
    if len(X) < 10:
        print("[ERROR] Dataset too small or missing target values.")
        return

    # Train / Val / Test split: 80% / 10% / 10%
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    print(f"      Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    print("\n[2/4] Initializing CatBoost Model...")
    model = CatBoostRegressor(
        iterations=args.iterations,
        learning_rate=args.lr,
        depth=args.depth,
        loss_function="RMSE",
        eval_metric="RMSE",
        cat_features=CATEGORICAL_COLS,
        verbose=100,
        random_seed=42,
        task_type="CPU"
    )

    train_pool = Pool(X_train, y_train, cat_features=CATEGORICAL_COLS)
    val_pool = Pool(X_val, y_val, cat_features=CATEGORICAL_COLS)

    print(f"\n[3/4] Training CatBoost for up to {args.iterations} iterations...")
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=50,
        use_best_model=True
    )

    print("\n[4/4] Evaluating on test set...")
    test_preds = model.predict(X_test)
    rmse, mae, r2, mape = metrics_str(test_preds, y_test.values)
    
    print("\n" + "="*40)
    print("CATBOOST TEST SET RESULTS (original scale)")
    print("="*40)
    print(f"RMSE  : {rmse:>12.2f}  g CO2")
    print(f"MAE   : {mae:>12.2f}  g CO2")
    print(f"R2    : {r2:>12.4f}")
    print(f"MAPE  : {mape:>11.2f}%")
    print("="*40)

    # Save model
    model_path = str(out / "carbon_fusion_catboost.cbm")
    model.save_model(model_path)
    print(f"\n[INFO] Model saved -> {model_path}")

    # Plot
    plot_pred_vs_actual(test_preds, y_test.values,
                        save_path=str(out / "pred_vs_actual_catboost.png"))

if __name__ == "__main__":
    main()
