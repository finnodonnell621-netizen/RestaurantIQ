"""
RestaurantIQ - model training & evaluation

Trains a Random Forest regressor to predict daily ingredient usage,
benchmarks it against a moving-average baseline, and evaluates both on
held-out (unseen) weeks using mean absolute error (MAE).

Outputs:
  src/model.pkl        - trained RandomForestRegressor
  src/encoders.pkl      - ingredient name <-> code mapping + per-(ingredient,
                          day-of-week) average sales, used by app.py to build
                          features for future days
  src/metrics.json      - MAE for both models, for display on the dashboard
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ingredient", "date"]).copy()
    df["dow"] = df["date"].dt.dayofweek
    df["ing_code"] = df["ingredient"].astype("category").cat.codes
    df["lag_1"] = df.groupby("ingredient")["used"].shift(1)
    df["lag_7"] = df.groupby("ingredient")["used"].shift(7)
    return df


def main():
    df = pd.read_csv("data/inventory_data.csv", parse_dates=["date"])
    df = build_features(df)
    df = df.dropna(subset=["lag_1", "lag_7"])

    # time-based split: hold out the last 2 weeks per ingredient (never
    # shuffle a time series into train/test randomly)
    split_date = df["date"].max() - pd.Timedelta(days=14)
    train, test = df[df["date"] <= split_date], df[df["date"] > split_date]

    features = ["dow", "ing_code", "lag_1", "lag_7", "sales"]
    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    model.fit(train[features], train["used"])
    rf_preds = model.predict(test[features])
    rf_mae = mean_absolute_error(test["used"], rf_preds)

    # baseline: predict the ingredient's trailing 7-day average usage
    baseline_preds = test.apply(
        lambda r: train[(train["ingredient"] == r["ingredient"]) & (train["date"] < r["date"])]
        .tail(7)["used"].mean(),
        axis=1,
    )
    baseline_preds = baseline_preds.fillna(train["used"].mean())
    baseline_mae = mean_absolute_error(test["used"], baseline_preds)

    print(f"Random Forest MAE : {rf_mae:.2f} lbs")
    print(f"Moving Avg MAE    : {baseline_mae:.2f} lbs")
    improvement = (baseline_mae - rf_mae) / baseline_mae * 100
    print(f"Improvement over baseline: {improvement:.1f}%")

    joblib.dump(model, "src/model.pkl")

    ingredient_codes = dict(enumerate(df["ingredient"].astype("category").cat.categories))
    code_lookup = {v: k for k, v in ingredient_codes.items()}
    # average sales by (ingredient, day-of-week), used as a stand-in for
    # "future sales" when forecasting days we haven't observed yet
    sales_by_dow = df.groupby(["ingredient", "dow"])["sales"].mean().to_dict()

    joblib.dump({
        "ingredient_to_code": code_lookup,
        "code_to_ingredient": ingredient_codes,
        "sales_by_ingredient_dow": sales_by_dow,
        "features": features,
    }, "src/encoders.pkl")

    eval_df = test[["date", "ingredient", "used"]].copy()
    eval_df["random_forest_pred"] = rf_preds
    eval_df["moving_avg_pred"] = baseline_preds.values
    eval_df.to_csv("src/eval_predictions.csv", index=False)

    with open("src/metrics.json", "w") as f:
        json.dump({
            "random_forest_mae": round(rf_mae, 2),
            "moving_average_mae": round(baseline_mae, 2),
            "improvement_pct": round(improvement, 1),
            "test_rows": int(len(test)),
        }, f, indent=2)

    print("Saved src/model.pkl, src/encoders.pkl, src/metrics.json")


if __name__ == "__main__":
    main()
