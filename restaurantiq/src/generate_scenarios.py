"""
RestaurantIQ - demo scenario generator

Builds the same kind of history as generate_data.py, but engineers the
final few days for specific ingredients so each file demonstrates a
different, clearly visible outcome on the dashboard:

  - scenario_low_stock.csv   -> several ingredients show up RED (low)
  - scenario_overstocked.csv -> several ingredients show up well-stocked
                                  after an oversized delivery
  - scenario_mixed.csv       -> a realistic mix of all statuses at once
                                  (recommended for the live demo)

Run:  python src/generate_scenarios.py
"""

import numpy as np
import pandas as pd

np.random.seed(7)

INGREDIENTS = {
    "Chicken": 14, "Ground Beef": 10, "Tomatoes": 9, "Cheese": 6,
    "Potatoes": 18, "Lettuce": 7, "Onions": 8, "Rice": 11,
    "Shrimp": 5, "Bacon": 6, "Mozzarella": 7, "Flour": 12,
    "Bell Peppers": 5, "Salmon": 4, "Black Beans": 6,
}

WEEKDAY_MULTIPLIER = {0: 0.90, 1: 0.85, 2: 0.90, 3: 1.00, 4: 1.40, 5: 1.60, 6: 1.15}
DELIVERY_DAYS = {0, 3}
WEEKS_OF_HISTORY = 10
START_DATE = pd.Timestamp("2026-07-13")
PAR_DAYS = 8.5


def generate_base():
    """Same order-up-to-par logic as generate_data.py - a normal, stable
    history ending mid-cycle (Wednesday) with a healthy buffer."""
    dates = pd.date_range(START_DATE, periods=WEEKS_OF_HISTORY * 7 + 3)
    rows = []
    for ingredient, base in INGREDIENTS.items():
        par_level = base * PAR_DAYS
        inventory = par_level
        for d in dates:
            mult = WEEKDAY_MULTIPLIER[d.weekday()]
            noise = np.random.normal(1.0, 0.12)
            desired_used = max(0.0, round(base * mult * noise, 1))
            beginning = round(inventory, 1)
            if d.weekday() in DELIVERY_DAYS:
                shortfall = max(0.0, par_level - beginning)
                delivery = round(shortfall * np.random.uniform(0.9, 1.15), 1)
            else:
                delivery = 0.0
            available = beginning + delivery
            used = round(min(desired_used, available), 1)
            ending = round(available - used, 1)
            inventory = ending
            sales = int(round(used * np.random.uniform(9, 11)))
            rows.append({
                "date": d, "ingredient": ingredient, "day_of_week": d.day_name(),
                "beginning_inventory": beginning, "delivery": delivery, "used": used,
                "ending_inventory": ending, "sales": sales,
            })
    return pd.DataFrame(rows)


def apply_low_stock(df, ingredients, days_back=3):
    """Zero out recent deliveries and bump usage for the last few days,
    so these ingredients end up with low inventory relative to demand."""
    df = df.copy()
    last_date = df["date"].max()
    for ing in ingredients:
        mask = (df["ingredient"] == ing) & (df["date"] > last_date - pd.Timedelta(days=days_back))
        idx = df[mask].sort_values("date").index
        prior = df[(df["ingredient"] == ing) & (df["date"] <= last_date - pd.Timedelta(days=days_back))]
        inventory = prior.sort_values("date").iloc[-1]["ending_inventory"]
        base = INGREDIENTS[ing]
        for i in idx:
            d = df.loc[i, "date"]
            mult = WEEKDAY_MULTIPLIER[d.weekday()] * 1.25  # demand runs hot
            used = round(base * mult, 1)
            delivery = 0.0  # no delivery arrived in time
            beginning = round(inventory, 1)
            ending = max(0.0, round(beginning + delivery - used, 1))
            inventory = ending
            df.loc[i, ["beginning_inventory", "delivery", "used", "ending_inventory"]] = [beginning, delivery, used, ending]
    return df


def apply_overstocked(df, ingredients, days_back=2):
    """Give these ingredients an oversized recent delivery, so they end
    up well above what's needed."""
    df = df.copy()
    last_date = df["date"].max()
    for ing in ingredients:
        mask = (df["ingredient"] == ing) & (df["date"] > last_date - pd.Timedelta(days=days_back))
        idx = df[mask].sort_values("date").index
        prior = df[(df["ingredient"] == ing) & (df["date"] <= last_date - pd.Timedelta(days=days_back))]
        inventory = prior.sort_values("date").iloc[-1]["ending_inventory"]
        base = INGREDIENTS[ing]
        first = True
        for i in idx:
            d = df.loc[i, "date"]
            mult = WEEKDAY_MULTIPLIER[d.weekday()]
            used = round(base * mult, 1)
            delivery = round(base * 9, 1) if first else 0.0  # a much bigger delivery than usual
            first = False
            beginning = round(inventory, 1)
            ending = round(beginning + delivery - used, 1)
            inventory = ending
            df.loc[i, ["beginning_inventory", "delivery", "used", "ending_inventory"]] = [beginning, delivery, used, ending]
    return df


def main():
    low_stock_ings = ["Chicken", "Shrimp", "Salmon", "Bell Peppers", "Tomatoes"]
    overstock_ings = ["Rice", "Flour", "Black Beans", "Potatoes", "Cheese"]

    # --- Scenario 1: Low Stock ---
    df1 = apply_low_stock(generate_base(), low_stock_ings)
    df1.to_csv("data/scenario_low_stock.csv", index=False, date_format="%Y-%m-%d")

    # --- Scenario 2: Overstocked ---
    df2 = apply_overstocked(generate_base(), overstock_ings)
    df2.to_csv("data/scenario_overstocked.csv", index=False, date_format="%Y-%m-%d")

    # --- Scenario 3: Mixed (recommended for the live demo) ---
    df3 = generate_base()
    df3 = apply_low_stock(df3, low_stock_ings[:3])       # 3 ingredients run low
    df3 = apply_overstocked(df3, overstock_ings[:3])     # 3 ingredients overstocked
    # the remaining 9 ingredients stay in their normal, well-balanced state
    df3.to_csv("data/scenario_mixed.csv", index=False, date_format="%Y-%m-%d")

    for name, d in [("scenario_low_stock.csv", df1), ("scenario_overstocked.csv", df2), ("scenario_mixed.csv", df3)]:
        print(f"Wrote data/{name} ({len(d)} rows)")


if __name__ == "__main__":
    main()
