"""
RestaurantIQ - synthetic data generator

Creates data/inventory_data.csv: one row per ingredient per day, with
beginning inventory, deliveries (input), usage (output), ending inventory,
and sales.

Deliveries use "order-up-to-par" logic (the same idea a real restaurant
uses): on a delivery day, the restaurant orders enough to bring stock back
up toward a target par level, rather than a fixed amount. This keeps
inventory oscillating in a realistic band around actual demand instead of
drifting upward forever - which matters because the dashboard's
low/reorder/good signals only make sense if current stock and predicted
demand are on the same scale.

Built-in patterns:
  - weekend usage spikes (Fri/Sat busiest)
  - two scheduled delivery days per week (Mon/Thu), order-up-to-par
  - random day-to-day noise
  - two "shock" days (unusually high demand)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ingredient -> average daily usage in lbs
INGREDIENTS = {
    "Chicken": 14, "Ground Beef": 10, "Tomatoes": 9, "Cheese": 6,
    "Potatoes": 18, "Lettuce": 7, "Onions": 8, "Rice": 11,
    "Shrimp": 5, "Bacon": 6, "Mozzarella": 7, "Flour": 12,
    "Bell Peppers": 5, "Salmon": 4, "Black Beans": 6,
}

WEEKDAY_MULTIPLIER = {0: 0.90, 1: 0.85, 2: 0.90, 3: 1.00, 4: 1.40, 5: 1.60, 6: 1.15}
DELIVERY_DAYS = {0, 3}          # Monday & Thursday
SHOCK_DAYS = {"2026-07-04", "2026-08-15"}

WEEKS_OF_HISTORY = 10
START_DATE = pd.Timestamp("2026-06-22")

PAR_DAYS = 8.5   # target: keep ~8.5 days of average usage on hand after a delivery
                 # (comfortable buffer through the Fri-Sat-Sun high-usage stretch)


def generate():
    # +3 days so history ends on a Wednesday - mid-cycle (2 days after a
    # delivery), giving a typical "mostly fine, a couple low" snapshot
    # rather than landing right at the pre-delivery low point every time.
    dates = pd.date_range(START_DATE, periods=WEEKS_OF_HISTORY * 7 + 3)
    rows = []

    for ingredient, base in INGREDIENTS.items():
        par_level = base * PAR_DAYS
        inventory = par_level  # start at par

        for d in dates:
            mult = WEEKDAY_MULTIPLIER[d.weekday()]
            if d.strftime("%Y-%m-%d") in SHOCK_DAYS:
                mult *= 1.8

            noise = np.random.normal(1.0, 0.12)
            desired_used = max(0.0, round(base * mult * noise, 1))

            beginning = round(inventory, 1)

            if d.weekday() in DELIVERY_DAYS:
                # order up to par, with a little real-world noise
                shortfall = max(0.0, par_level - beginning)
                delivery = round(shortfall * np.random.uniform(0.9, 1.15), 1)
            else:
                delivery = 0.0

            available = beginning + delivery
            used = round(min(desired_used, available), 1)  # can't use what you don't have
            ending = round(available - used, 1)
            inventory = ending

            sales = int(round(used * np.random.uniform(9, 11)))

            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "ingredient": ingredient,
                "day_of_week": d.day_name(),
                "beginning_inventory": beginning,
                "delivery": delivery,
                "used": used,
                "ending_inventory": ending,
                "sales": sales,
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/inventory_data.csv", index=False)
    print(f"Wrote {len(df)} rows across {len(INGREDIENTS)} ingredients "
          f"({WEEKS_OF_HISTORY} weeks) to data/inventory_data.csv")


if __name__ == "__main__":
    generate()
