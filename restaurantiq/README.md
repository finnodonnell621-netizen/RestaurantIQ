# RestaurantIQ

AI-powered restaurant inventory system: predicts ingredient demand from
historical usage and sales, then recommends what to order and how much.

## What's already done for you

This project is fully built and already run once, so everything works out
of the box:

- `data/inventory_data.csv` - 10 weeks of synthetic data, 15 ingredients
- `data/scenario_*.csv` - 3 extra example datasets for demoing (see below)
- `src/model.pkl` - trained Random Forest model
- `src/metrics.json` / `src/eval_predictions.csv` - evaluation results
- `app.py` - the live dashboard

## Demo datasets

The dashboard has a sidebar dropdown ("Demo dataset") to switch between
four example scenarios live, without restarting anything:

| Dataset | What it shows |
|---|---|
| **Mixed — Realistic Demo** (default) | A realistic blend: a few ingredients low, a few overstocked, most fine — the best one to lead with live, since it shows every status at once |
| **Typical Day** | A normal, healthy day — mostly green, one or two needing a reorder soon |
| **Low Stock Day** | Several ingredients clearly running low, with real recommended order quantities |
| **Overstocked Day** | Several ingredients oversupplied after an oversized delivery, flagged to skip the next order |

All four use the same trained model — only the *current* inventory
snapshot changes, exactly like a real restaurant's data would look
different from one day to the next.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501 in your browser.

## Regenerate data or retrain (optional)

Only needed if you want different data/ingredients, or to tweak the model:

```bash
python src/generate_data.py        # rebuilds data/inventory_data.csv
python src/generate_scenarios.py   # rebuilds the 3 demo scenario files
python src/train_model.py          # retrains, prints MAE, saves model.pkl
```

## Project structure

```
restaurantiq/
├── data/inventory_data.csv     synthetic dataset
├── src/
│   ├── generate_data.py        builds the dataset
│   ├── train_model.py          trains + evaluates the model
│   ├── model.pkl                trained Random Forest
│   ├── encoders.pkl             ingredient codes + sales lookup for forecasting
│   ├── metrics.json             MAE results (RF vs. moving average)
│   └── eval_predictions.csv     predicted vs. actual on held-out weeks
├── app.py                       Streamlit dashboard
└── requirements.txt
```

## How it works

1. **Data**: one row per ingredient per day — beginning inventory, delivery
   (input), usage (output), ending inventory, sales. Weekend usage spikes
   and two "shock" days are built in.
2. **Model**: a Random Forest Regressor trained across all ingredients
   (ingredient + day-of-week + recent usage + sales as features),
   benchmarked against a moving-average baseline. Evaluated on 2 held-out
   weeks using mean absolute error (MAE) — last run: see `src/metrics.json`.
3. **Dashboard**: for each ingredient, forecasts the next 3 days of usage,
   compares it to current stock, flags low/overstocked items, and
   recommends an order quantity (with a 10% safety buffer).

## Known limitations (for the ethics/limitations slide)

- The model doesn't know about one-off events (large reservations,
  holidays, promotions) unless they're reflected in past data — it
  supports the manager's judgment, it doesn't replace it.
- Forecasts assume next-week sales resemble the historical average for
  that day of week, since actual future sales aren't known in advance.
- Accuracy depends on inventory being counted/entered correctly — bad
  data in, bad recommendations out.
