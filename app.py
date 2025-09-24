
# app.py — Melbourne House Price Estimator (Pro dashboard)
import json, math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import altair as alt

# ============================ PAGE & THEME =============================
st.set_page_config(
    page_title="Melbourne House Price Estimator",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Subtle modern styling
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px;}
    .metric-card {
        background: linear-gradient(145deg, #1f2937 0%, #0f172a 100%);
        border: 1px solid #334155; border-radius: 14px; padding: 16px; color: #e5e7eb;
    }
    .metric-card h3 {margin: 0 0 6px 0; font-weight: 600; font-size: 0.95rem; color: #a5b4fc;}
    .metric-card .value {font-size: 1.6rem; font-weight: 700;}
    .subtle {color:#94a3b8;}
    .pill {
        display:inline-block; padding:4px 10px; border-radius:999px; background:#0ea5e9; color:white; font-size:0.75rem;
        margin-left:8px;
    }
    .card {border: 1px solid #334155; border-radius: 12px; padding: 14px;}
    .em {color:#a5b4fc; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ============================== HELPERS ================================
@st.cache_resource
def load_model(path="model.joblib"):
    return joblib.load(path)

@st.cache_data
def load_clean_data(path="realestate_clean.csv"):
    p = Path(path)
    if not p.exists(): return None
    try:
        return pd.read_csv(p, parse_dates=["sold_date"])
    except Exception:
        return pd.read_csv(p)

def make_row(suburb, ptype, beds, baths, cars, land, dist, sold_y, sold_m):
    # Build a single-row DataFrame with the columns your pipeline expects.
    return pd.DataFrame([{
        "bedrooms": beds, "bathrooms": baths, "car_spaces": cars, "land_m2": land,
        "property_type": ptype, "suburb": suburb, "postcode": np.nan,
        "lat": np.nan, "lon": np.nan, "sold_year": sold_y, "sold_month": sold_m,
        "dist_to_cbd_km": dist, "price_per_m2": np.nan   # imputed inside pipeline
    }])

def compute_conformal_q(model, df_clean, alpha=0.1):
    """
    Split-conformal-style quantile on LOG residuals from the cleaned dataset (if available).
    For 'proper' intervals use a held-out calibration split during training and save q.
    """
    if df_clean is None or "price_target" not in df_clean.columns:
        return None
    need = ["bedrooms","bathrooms","car_spaces","land_m2","property_type","suburb","postcode",
            "lat","lon","sold_year","sold_month","dist_to_cbd_km","price_per_m2"]
    X_train = df_clean[[c for c in need if c in df_clean.columns]].copy()
    for c in need:
        if c not in X_train.columns:
            X_train[c] = np.nan
    y_log = np.log1p(df_clean["price_target"].values)
    yhat_log = model.predict(X_train[need])
    resid = np.abs(y_log - yhat_log)
    # alpha here is miscoverage (e.g., 0.1 → 90% interval)
    q = float(np.quantile(resid, 1 - alpha))
    return q

def interval_from_log(y_log, q):
    pt = np.expm1(y_log)
    if q is None: return pt, None, None
    return pt, np.expm1(y_log - q), np.expm1(y_log + q)

def plot_distribution(series, vline=None, title="", xlab="", fmt="${,.0f}"):
    if series is None or len(series)==0: 
        st.info("Not enough comparable data to plot.")
        return
    df = pd.DataFrame({"value": series})
    chart = (alt.Chart(df)
             .transform_density("value", as_=["value","density"])
             .mark_area(opacity=0.5)
             .encode(x=alt.X("value:Q", title=xlab),
                     y=alt.Y("density:Q", title="Density"),
                     tooltip=[alt.Tooltip("value:Q", format=fmt)])
             .properties(height=220, title=title))
    if vline is not None:
        v = pd.DataFrame({"v":[vline]})
        rule = alt.Chart(v).mark_rule(color="#f59e0b", strokeWidth=2).encode(x="v:Q")
        chart = chart + rule
    st.altair_chart(chart, use_container_width=True)

# =========================== LOAD ARTIFACTS ============================
model = load_model("model.joblib")
df_clean = load_clean_data("realestate_clean.csv")

# Load or compute conformal q
q = None
try:
    if Path("conformal_q.json").exists():
        q = float(json.loads(Path("conformal_q.json").read_text())["q_log"])
    elif Path("conformal_q.npy").exists():
        q = float(np.load("conformal_q.npy"))
except Exception:
    q = None
default_alpha = 0.10
if q is None:
    q = compute_conformal_q(model, df_clean, alpha=default_alpha)

# ================================ UI ==================================
st.markdown("<h1>🏡 Melbourne House Price Estimator <span class='pill'>Pro</span></h1>", unsafe_allow_html=True)
st.caption("Trained ML pipeline with optional conformal prediction intervals, local comps, and what-if analysis.")

# Sidebar controls
st.sidebar.header("Controls")
ui_mode = st.sidebar.radio("Mode", ["Single estimate", "Compare scenarios", "What-if"])
alpha = st.sidebar.slider("Interval miscoverage α (lower = wider interval)", 0.02, 0.30, value=0.10, step=0.01)
use_interval = st.sidebar.checkbox("Show prediction interval", value=(q is not None))
show_comps = st.sidebar.checkbox("Show local comps (if data available)", value=True)
show_ppm2 = st.sidebar.checkbox("Show price-per-m² visuals", value=True)

# Recompute q to match chosen alpha if we have df_clean
q_dynamic = compute_conformal_q(model, df_clean, alpha=alpha) if (use_interval and df_clean is not None) else (q if use_interval else None)

# Suburb list if available
suburb_options = None
if df_clean is not None and "suburb" in df_clean.columns:
    try:
        suburb_options = sorted([s for s in df_clean["suburb"].dropna().unique().tolist() if isinstance(s, str) and s.strip() != ""])
    except Exception:
        suburb_options = None

ptype_options = ["house","unit","townhouse","apartment","villa","other"]

# -------------------------- INPUT COMPONENTS ---------------------------
def scenario_inputs(key=""):
    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        suburb = (st.selectbox("Suburb", suburb_options, index=0, key=f"suburb{key}")
                  if suburb_options else st.text_input("Suburb", "Glen Waverley", key=f"suburb{key}"))
        ptype = st.selectbox("Property type", ptype_options, index=0, key=f"ptype{key}")
        beds  = st.number_input("Bedrooms", 0, 12, 3, key=f"beds{key}")
        baths = st.number_input("Bathrooms", 0, 12, 2, key=f"baths{key}")
    with c2:
        cars  = st.number_input("Car spaces", 0, 10, 1, key=f"cars{key}")
        land  = st.number_input("Land size (m²)", 0, 20000, 500, key=f"land{key}")
        dist  = st.number_input("Distance to CBD (km)", 0.0, 60.0, 18.0, key=f"dist{key}")
    with c3:
        sold_y= st.number_input("Sold year", 2000, 2035, 2025, key=f"sy{key}")
        sold_m= st.number_input("Sold month", 1, 12, 9, key=f"sm{key}")
    return suburb, ptype, beds, baths, cars, land, dist, sold_y, sold_m

# =============================== MODES =================================
if ui_mode == "Single estimate":
    vals = scenario_inputs("")
    if st.button("Predict price", type="primary"):
        X_row = make_row(*vals)
        y_log = model.predict(X_row)[0]
        pred, lo, hi = interval_from_log(y_log, q_dynamic)

        # Metric cards
        m1, m2, m3 = st.columns(3)
        with m1: st.markdown(f"<div class='metric-card'><h3>Estimated price</h3><div class='value'>${pred:,.0f}</div></div>", unsafe_allow_html=True)
        with m2:
            if lo is not None:
                st.markdown(f"<div class='metric-card'><h3>90% lower</h3><div class='value'>${lo:,.0f}</div></div>", unsafe_allow_html=True)
            else:
                st.empty()
        with m3:
            if hi is not None:
                st.markdown(f"<div class='metric-card'><h3>90% upper</h3><div class='value'>${hi:,.0f}</div></div>", unsafe_allow_html=True)
            else:
                st.empty()

        # Derived metric
        if vals[5] and vals[5] > 0:
            ppm2 = pred / vals[5]
            st.markdown(f"<span class='subtle'>Approx. price per m²: </span><span class='em'>${ppm2:,.0f}/m²</span>", unsafe_allow_html=True)

        # Local comps (if data)
        if show_comps and df_clean is not None:
            st.subheader("Local comps")
            # Filter comps by suburb, property type, +/- 1 bedroom (if available)
            comps = df_clean.copy()
            comps = comps[comps["suburb"].astype(str).str.lower() == str(vals[0]).lower()]
            comps = comps[comps["property_type"].astype(str).str.lower() == str(vals[1]).lower()]
            if "bedrooms" in comps.columns and pd.notna(vals[2]):
                comps = comps[(comps["bedrooms"] >= vals[2]-1) & (comps["bedrooms"] <= vals[2]+1)]
            # Show price distribution
            if "price_target" in comps.columns and len(comps) >= 10:
                plot_distribution(comps["price_target"].dropna().values, vline=pred,
                                  title="Sale price distribution — comparable set", xlab="Sale price ($)")
            else:
                st.info("Not enough comparable sales to show a distribution for this selection.")
            # Optional price-per-m² distribution
            if show_ppm2 and "price_per_m2" in comps.columns:
                comp_ppm2 = comps["price_per_m2"].replace([np.inf, -np.inf], np.nan).dropna()
                if len(comp_ppm2) >= 10 and vals[5] and vals[5] > 0:
                    plot_distribution(comp_ppm2.values, vline=(pred/vals[5]),
                                      title="Price-per-m² — comparable set", xlab="Price per m² ($)")
        # Download one-liner report
        row = dict(zip(
            ["suburb","ptype","bedrooms","bathrooms","car_spaces","land_m2","dist_to_cbd_km","sold_year","sold_month"],
            vals
        ))
        row.update({"pred_$": float(pred), "lo_$": float(lo) if lo is not None else None, "hi_$": float(hi) if hi is not None else None})
        rep = pd.DataFrame([row]).to_csv(index=False)
        st.download_button("Download prediction (.csv)", rep, file_name="prediction.csv", mime="text/csv")

elif ui_mode == "Compare scenarios":
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Scenario A**"); A = scenario_inputs("A")
    with cB:
        st.markdown("**Scenario B**"); B = scenario_inputs("B")
    if st.button("Compare", type="primary"):
        XA, XB = make_row(*A), make_row(*B)
        yA, yB = model.predict(XA)[0], model.predict(XB)[0]
        A_pred, A_lo, A_hi = interval_from_log(yA, q_dynamic)
        B_pred, B_lo, B_hi = interval_from_log(yB, q_dynamic)

        # Metric cards
        m1, m2, m3 = st.columns(3)
        with m1: st.markdown(f"<div class='metric-card'><h3>Scenario A</h3><div class='value'>${A_pred:,.0f}</div></div>", unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-card'><h3>Scenario B</h3><div class='value'>${B_pred:,.0f}</div></div>", unsafe_allow_html=True)
        delta = B_pred - A_pred
        with m3: st.markdown(f"<div class='metric-card'><h3>Δ (B − A)</h3><div class='value'>{delta:+,.0f} $</div></div>", unsafe_allow_html=True)

        # Visual compare
        plot_df = pd.DataFrame({
            "Scenario": ["A","B"],
            "Price": [A_pred, B_pred]
        })
        bar = alt.Chart(plot_df).mark_bar().encode(
            x=alt.X("Scenario:N", title=""),
            y=alt.Y("Price:Q", title="Predicted price ($)"),
            color=alt.Color("Scenario:N", scale=alt.Scale(range=["#60a5fa","#34d399"])),
            tooltip=[alt.Tooltip("Price:Q", format="${,.0f}")]
        ).properties(height=250)
        st.altair_chart(bar, use_container_width=True)

        # Interval table
        if q_dynamic is not None:
            st.dataframe(pd.DataFrame({
                "Scenario":["A","B"],
                "Predicted $":[A_pred,B_pred],
                "Lo (90%)":[A_lo,B_lo],
                "Hi (90%)":[A_hi,B_hi]
            }).style.format({"Predicted $":"${:,.0f}","Lo (90%)":"${:,.0f}","Hi (90%)":"${:,.0f}"}), use_container_width=True)

else:  # What-if
    st.markdown("### What-if analysis")
    base = scenario_inputs("W")
    st.markdown("<div class='subtle'>Change one feature while holding others constant to see sensitivity.</div>", unsafe_allow_html=True)

    what = st.selectbox("Vary which feature?", ["Distance to CBD (km)","Land size (m²)","Bedrooms"])
    steps = st.slider("Number of steps", 10, 60, 25)
    if st.button("Run what-if", type="primary"):
        if what == "Distance to CBD (km)":
            xs = np.linspace(0, 40, steps)
            rows = [make_row(base[0], base[1], base[2], base[3], base[4], base[5], x, base[7], base[8]) for x in xs]
            xlab = "Distance to CBD (km)"
        elif what == "Land size (m²)":
            lo = max(100, base[5]*0.5); hi = max(200, base[5]*1.5)
            xs = np.linspace(lo, hi, steps)
            rows = [make_row(base[0], base[1], base[2], base[3], base[4], l, base[6], base[7], base[8]) for l in xs]
            xlab = "Land size (m²)"
        else:  # Bedrooms
            lo = max(0, base[2]-2); hi = min(8, base[2]+2)
            xs = np.arange(lo, hi+1)
            rows = [make_row(base[0], base[1], b, base[3], base[4], base[5], base[6], base[7], base[8]) for b in xs]
            xlab = "Bedrooms"

        X_grid = pd.concat(rows, ignore_index=True)
        yhat = np.expm1(model.predict(X_grid))
        df_plot = pd.DataFrame({xlab: xs, "Predicted $": yhat})
        line = alt.Chart(df_plot).mark_line(point=True).encode(
            x=alt.X(f"{xlab}:Q", title=xlab),
            y=alt.Y("Predicted $:Q", title="Predicted price ($)"),
            tooltip=[alt.Tooltip("Predicted $:Q", format="${,.0f}"), xlab]
        ).properties(height=280)
        st.altair_chart(line, use_container_width=True)

# ============================== FOOTER =================================
st.markdown("---")
st.caption(
    "Intervals use split-conformal residuals computed from available data. "
    "This dashboard uses a trained pipeline (preprocessing + model). "
    "Educational demo; not financial advice."
)
