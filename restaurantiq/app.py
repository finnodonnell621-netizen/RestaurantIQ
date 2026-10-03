"""
RestaurantIQ Dashboard

Run with:  streamlit run app.py

Loads the trained model + the ingredient data, forecasts each ingredient's
demand over the next 3 days, flags low/overstocked items, and recommends
an order quantity - the live demo screen for the presentation, and the
day-to-day screen a restaurant manager would actually use.
"""

import json

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="RestaurantIQ", page_icon="🍽️", layout="wide")

DATASETS = {
    "Mixed — Realistic Demo": "data/scenario_mixed.csv",
    "Typical Day": "data/inventory_data.csv",
    "Low Stock Day": "data/scenario_low_stock.csv",
    "Overstocked Day": "data/scenario_overstocked.csv",
}
FORECAST_DAYS = 3
SAFETY_BUFFER_PCT = 0.10

STATUS_COLOR = {"Low": "#B3492A", "Reorder Soon": "#B08A1E", "Overstocked": "#2A5DA8", "Good": "#2E6B47"}
STATUS_BG = {"Low": "#FBEDE7", "Reorder Soon": "#FBF3DE", "Overstocked": "#E8EEF9", "Good": "#EAF3EC"}

st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 1100px;}
    h1 {font-size: 2.1rem !important;}
    .rq-card {
        border-radius: 12px; padding: 18px 20px; margin-bottom: 12px;
        border-left: 6px solid; display: flex; justify-content: space-between;
        align-items: center; flex-wrap: wrap; gap: 8px;
    }
    .rq-card .name {font-size: 1.05rem; font-weight: 700;}
    .rq-card .sub {font-size: 0.85rem; opacity: 0.75;}
    .rq-card .order {font-size: 1.4rem; font-weight: 800; text-align: right;}
    .rq-card .order-label {font-size: 0.75rem; font-weight: 500; opacity: 0.75; text-align: right;}
    .status-dot {
        display: inline-block; width: 44px; height: 44px; border-radius: 50%;
        text-align: center; line-height: 44px; font-size: 1.3rem; margin-bottom: 6px;
    }
    .status-grid-item {text-align: center; padding: 4px;}
    .status-grid-item .ing-name {font-size: 0.85rem; font-weight: 600;}
    .status-grid-item .ing-status {font-size: 0.72rem; opacity: 0.7;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_encoders():
    model = joblib.load("src/model.pkl")
    encoders = joblib.load("src/encoders.pkl")
    with open("src/metrics.json") as f:
        metrics = json.load(f)
    return model, encoders, metrics


@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["ingredient", "date"])


def forecast_ingredient(model, encoders, df, ingredient, days=FORECAST_DAYS):
    """Recursively predict the next `days` of usage for one ingredient."""
    hist = df[df["ingredient"] == ingredient].sort_values("date")
    ing_code = encoders["ingredient_to_code"][ingredient]
    last_date = hist["date"].max()
    last_used = hist.iloc[-1]["used"]

    predictions = []
    lag_1 = last_used
    for step in range(1, days + 1):
        target_date = last_date + pd.Timedelta(days=step)
        dow = target_date.dayofweek

        seven_back = target_date - pd.Timedelta(days=7)
        row_7 = hist[hist["date"] == seven_back]
        lag_7 = row_7["used"].values[0] if len(row_7) else hist["used"].tail(7).mean()

        sales_proxy = encoders["sales_by_ingredient_dow"].get((ingredient, dow), hist["sales"].mean())

        features = pd.DataFrame([{
            "dow": dow, "ing_code": ing_code, "lag_1": lag_1,
            "lag_7": lag_7, "sales": sales_proxy,
        }])[encoders["features"]]

        pred = max(0.0, model.predict(features)[0])
        predictions.append(pred)
        lag_1 = pred

    return sum(predictions)


def status_for(current, predicted):
    if current < predicted:
        return "Low", "🔴"
    if current < predicted * 1.2:
        return "Reorder Soon", "🟡"
    if current > predicted * 2:
        return "Overstocked", "🔵"
    return "Good", "🟢"


@st.cache_data
def compute_results(_model, _encoders, df):
    latest = df.groupby("ingredient").tail(1).set_index("ingredient")
    rows = []
    for ing in sorted(df["ingredient"].unique()):
        current = latest.loc[ing, "ending_inventory"]
        predicted = forecast_ingredient(_model, _encoders, df, ing)
        recommended = max(0.0, (predicted - current) + predicted * SAFETY_BUFFER_PCT)
        label, emoji = status_for(current, predicted)
        rows.append({
            "ingredient": ing, "current": current, "predicted": predicted,
            "recommended": recommended, "label": label, "emoji": emoji,
        })
    order = {"Low": 0, "Reorder Soon": 1, "Overstocked": 2, "Good": 3}
    return pd.DataFrame(rows).sort_values(by="label", key=lambda s: s.map(order)).reset_index(drop=True)


with st.sidebar:
    st.subheader("🍽️ RestaurantIQ")
    dataset_label = st.selectbox("Demo dataset", list(DATASETS.keys()))
    st.caption(
        "Switch between example scenarios to see how the dashboard reacts: "
        "a typical day, a low-stock day, an overstocked day, or a realistic mix of all three."
    )

model, encoders, metrics = load_model_and_encoders()
df = load_data(DATASETS[dataset_label])
results_df = compute_results(model, encoders, df)

st.title("🍽️ RestaurantIQ")
st.caption("What to order, how much, and when — based on your recent usage.")
st.write("")

needs_order = results_df[results_df["label"].isin(["Low", "Reorder Soon"])]
overstocked = results_df[results_df["label"] == "Overstocked"]

st.subheader("Needs Your Attention")
if needs_order.empty and overstocked.empty:
    st.success("Everything is well-stocked. No orders needed right now.")
else:
    for _, r in needs_order.iterrows():
        color, bg = STATUS_COLOR[r["label"]], STATUS_BG[r["label"]]
        st.markdown(f"""
        <div class="rq-card" style="border-color:{color}; background:{bg};">
            <div>
                <div class="name">{r['emoji']} {r['ingredient']}</div>
                <div class="sub">{r['current']:.0f} lbs on hand &middot; {r['predicted']:.0f} lbs needed over next {FORECAST_DAYS} days</div>
            </div>
            <div>
                <div class="order" style="color:{color};">{r['recommended']:.0f} lbs</div>
                <div class="order-label">order now</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    for _, r in overstocked.iterrows():
        color, bg = STATUS_COLOR[r["label"]], STATUS_BG[r["label"]]
        excess = r["current"] - r["predicted"]
        st.markdown(f"""
        <div class="rq-card" style="border-color:{color}; background:{bg};">
            <div>
                <div class="name">{r['emoji']} {r['ingredient']}</div>
                <div class="sub">{r['current']:.0f} lbs on hand &middot; only {r['predicted']:.0f} lbs needed over next {FORECAST_DAYS} days</div>
            </div>
            <div>
                <div class="order" style="color:{color};">{excess:.0f} lbs</div>
                <div class="order-label">surplus — skip next order</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.write("")
st.subheader("Full Inventory Status")
cols = st.columns(5)
for i, (_, r) in enumerate(results_df.iterrows()):
    with cols[i % 5]:
        st.markdown(f"""
        <div class="status-grid-item">
            <div class="status-dot" style="background:{STATUS_BG[r['label']]};">{r['emoji']}</div>
            <div class="ing-name">{r['ingredient']}</div>
            <div class="ing-status">{r['label']}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")
st.divider()

st.subheader("Ingredient Detail")
selected = st.selectbox("Select an ingredient", results_df["ingredient"], label_visibility="collapsed")
row = results_df[results_df["ingredient"] == selected].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Inventory", f"{row['current']:.0f} lbs")
c2.metric(f"Predicted {FORECAST_DAYS}-day Demand", f"{row['predicted']:.0f} lbs")
c3.metric("Status", f"{row['emoji']} {row['label']}")
c4.metric("Recommended Order", f"{row['recommended']:.0f} lbs")

hist = df[df["ingredient"] == selected].tail(21)
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["date"], y=hist["used"], mode="lines+markers", name="Daily usage",
                          line=dict(color="#2E6B47")))
fig.add_trace(go.Scatter(x=hist["date"], y=hist["ending_inventory"], mode="lines", name="Ending inventory",
                          line=dict(color="#B08A1E", dash="dot"), yaxis="y2"))
fig.update_layout(
    yaxis=dict(title="lbs used/day"),
    yaxis2=dict(title="lbs on hand", overlaying="y", side="right"),
    legend=dict(orientation="h", y=1.15),
    height=340, margin=dict(l=10, r=10, t=20, b=10),
    plot_bgcolor="white",
)
st.plotly_chart(fig, width="stretch")

st.write("")
with st.expander("Full recommendation table"):
    st.dataframe(
        results_df.rename(columns={
            "ingredient": "Ingredient", "current": "Current (lbs)",
            "predicted": f"Predicted {FORECAST_DAYS}-day Demand (lbs)",
            "recommended": "Recommended Order (lbs)", "label": "Status",
        })[["Ingredient", "Current (lbs)", f"Predicted {FORECAST_DAYS}-day Demand (lbs)",
            "Recommended Order (lbs)", "Status"]].round(1),
        width="stretch", hide_index=True,
    )

with st.expander("Model accuracy (technical detail)"):
    m1, m2, m3 = st.columns(3)
    m1.metric("Random Forest MAE", f"{metrics['random_forest_mae']} lbs")
    m2.metric("Moving Average MAE", f"{metrics['moving_average_mae']} lbs")
    m3.metric("Improvement over baseline", f"{metrics['improvement_pct']}%")
    st.caption("Mean absolute error (MAE): average difference between predicted and actual usage, "
               "measured on weeks of data the model never trained on.")

st.write("")
st.caption(
    "Recommendations support the manager's judgment — they don't replace it. "
    "The model can't see one-off events like large reservations, holidays, or promotions."
)
