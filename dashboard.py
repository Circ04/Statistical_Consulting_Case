"""
ORTEC Dynamic Pricing — Decision Dashboard
==========================================

Streamlit app that wraps the Bayesian model output in a pricing-manager
friendly interface.  Three layers, exactly as the strategy document
prescribes:

    Layer 1 — Executive summary  (the euro question)
    Layer 2 — Decision support   (per-ROMGID flags, where to focus)
    Layer 3 — Strategic insight  (elasticity by segment, scenarios)

Run with:
    streamlit run dashboard.py -- --recommendations outputs/recommendations.csv \\
                                  --idata outputs/idata.nc

If the CSV/NetCDF files are absent the app falls back to synthetic demo
data so reviewers can click around without a fitted model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — runs once
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ORTEC Dynamic Pricing",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root {
    --ink: #1A1F2B;
    --paper: #FAFAF7;
    --line: #E8E5DC;
    --accent: #C04848;
    --green: #4C9A4A;
    --amber: #E0B341;
    --blue: #2E5C8A;
}
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
}
h1, h2, h3 {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 500;
    letter-spacing: -0.01em;
}
.stMetric {
    background: white;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1rem;
}
[data-testid="stMetricLabel"] {
    color: #6B6962;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 500;
    color: var(--ink);
}
.flag-green { color: var(--green); font-weight: 600; }
.flag-amber { color: var(--amber); font-weight: 600; }
.flag-red { color: var(--accent); font-weight: 600; }
.section-divider {
    border-top: 1px solid var(--line);
    margin: 2rem 0 1.5rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading — with synthetic fallback
# ---------------------------------------------------------------------------

@st.cache_data
def load_recommendations(path: str | None) -> pd.DataFrame:
    """Load the recommendations CSV; fall back to synthetic data."""
    if path and Path(path).exists():
        return pd.read_csv(path, parse_dates=["WeekStartDate"])
    return _synthetic_recommendations()


def _synthetic_recommendations() -> pd.DataFrame:
    """Realistic-ish dummy data so the dashboard works without a fitted model."""
    rng = np.random.default_rng(42)
    n = 2000
    campsites = [f"CAMP_{i:03d}" for i in range(40)]
    markets = ["BE_FR", "NL", "DE", "FR_FR", "UK"]
    types = ["Comfort", "Premium", "Budget", "Family"]
    df = pd.DataFrame({
        "CampsiteCode":      rng.choice(campsites, n),
        "MarketGroupCode":   rng.choice(markets, n),
        "AccoTypeRangeCode": rng.choice(types, n),
        "WeekStartDate":     pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="d"),
        "p_obs_mean":        rng.uniform(45, 220, n),
        "TBN":               rng.poisson(15, n),
        "capacity":          rng.integers(20, 60, n),
    })
    df["p_obs_min"] = df["p_obs_mean"] * rng.uniform(0.85, 0.98, n)
    df["p_obs_max"] = df["p_obs_mean"] * rng.uniform(1.02, 1.18, n)
    df["p_obs_std"] = df["p_obs_mean"] * rng.uniform(0.02, 0.20, n)
    df["recommended_price"]  = 1.15 * df["p_obs_max"]
    df["price_change_pct"]   = 100 * (df["recommended_price"] / df["p_obs_mean"] - 1)
    beta = -0.75
    df["expected_bookings"]        = df["TBN"] * (df["recommended_price"] / df["p_obs_mean"]) ** beta
    df["expected_bookings_capped"] = np.minimum(df["expected_bookings"], df["capacity"])
    df["capacity_binds"]           = df["expected_bookings"] > df["capacity"]
    df["observed_revenue"]         = df["p_obs_mean"] * df["TBN"]
    df["expected_revenue"]         = df["recommended_price"] * df["expected_bookings_capped"]
    df["uplift_eur"]               = df["expected_revenue"] - df["observed_revenue"]
    df["uplift_pct"] = np.where(df["observed_revenue"] > 0,
                                100 * df["uplift_eur"] / df["observed_revenue"], np.nan)
    cv = df["p_obs_std"] / df["p_obs_mean"]
    df["hitl_flag"] = np.select(
        [(df["price_change_pct"] > 25) | (cv > 0.30),
         (df["price_change_pct"] > 15) | (cv > 0.15)],
        ["red", "yellow"], default="green",
    )
    return df


@st.cache_data
def load_posterior(path: str | None) -> dict:
    """Load posterior summary; falls back to synthetic posterior."""
    if path and Path(path).exists():
        try:
            import arviz as az
            idata = az.from_netcdf(path)
            beta_post = idata.posterior["log_price_z"].values.flatten()
            return {
                "beta_samples": beta_post,
                "beta_mean":    float(beta_post.mean()),
                "beta_lo":      float(np.percentile(beta_post, 10)),
                "beta_hi":      float(np.percentile(beta_post, 90)),
            }
        except Exception:
            pass
    rng = np.random.default_rng(42)
    samples = rng.normal(-0.752, 0.090, 2000)
    return {
        "beta_samples": samples,
        "beta_mean":    float(samples.mean()),
        "beta_lo":      float(np.percentile(samples, 10)),
        "beta_hi":      float(np.percentile(samples, 90)),
    }


# ---------------------------------------------------------------------------
# Argparse — only when running outside Streamlit Cloud
# ---------------------------------------------------------------------------

def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--recommendations", default=None)
    p.add_argument("--idata",           default=None)
    args, _ = p.parse_known_args()
    return args

cli   = parse_cli()
df    = load_recommendations(cli.recommendations)
post  = load_posterior(cli.idata)


# ---------------------------------------------------------------------------
# Sidebar — global filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Filters")

    markets = st.multiselect(
        "Market groups",
        options=sorted(df["MarketGroupCode"].unique()),
        default=sorted(df["MarketGroupCode"].unique()),
    )
    types = st.multiselect(
        "Accommodation types",
        options=sorted(df["AccoTypeRangeCode"].unique()),
        default=sorted(df["AccoTypeRangeCode"].unique()),
    )
    flag_filter = st.multiselect(
        "HITL flag",
        options=["green", "yellow", "red"],
        default=["green", "yellow", "red"],
    )
    week_range = st.date_input(
        "Stay-week range",
        value=(df["WeekStartDate"].min().date(), df["WeekStartDate"].max().date()),
    )

    st.markdown("---")
    st.markdown("### Adoption scenario")
    adoption = st.select_slider(
        "Which flags do we accept automatically?",
        options=["green only", "green + yellow", "all"],
        value="green + yellow",
    )

mask = (
    df["MarketGroupCode"].isin(markets)
    & df["AccoTypeRangeCode"].isin(types)
    & df["hitl_flag"].isin(flag_filter)
)
if isinstance(week_range, tuple) and len(week_range) == 2:
    mask &= (df["WeekStartDate"] >= pd.Timestamp(week_range[0])) & \
            (df["WeekStartDate"] <= pd.Timestamp(week_range[1]))
fdf = df[mask].copy()

flags_in_scope = {"green only": ["green"],
                  "green + yellow": ["green", "yellow"],
                  "all": ["green", "yellow", "red"]}[adoption]
adopted = fdf[fdf["hitl_flag"].isin(flags_in_scope)]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("ORTEC Dynamic Pricing")
st.caption("Bayesian hierarchical pricing recommendations — KU Leuven Statistical Consulting 2025–2026")


# ---------------------------------------------------------------------------
# Layer 1 — Executive summary
# ---------------------------------------------------------------------------

st.markdown("## Executive summary")

c1, c2, c3, c4 = st.columns(4)
total_uplift   = adopted["uplift_eur"].sum()
adopted_revenue = adopted["expected_revenue"].sum()
baseline_revenue = adopted["observed_revenue"].sum()
n_decisions    = len(fdf)
n_adopted      = len(adopted)
beta_mean      = post["beta_mean"]
beta_lo, beta_hi = post["beta_lo"], post["beta_hi"]

c1.metric(
    "Expected uplift",
    f"€{total_uplift:,.0f}",
    f"{(total_uplift / baseline_revenue * 100 if baseline_revenue > 0 else 0):+.1f}%",
)
c2.metric(
    "Decisions in scope",
    f"{n_decisions:,}",
    f"{n_adopted:,} adopted under '{adoption}'",
)
c3.metric(
    "Global elasticity β",
    f"{beta_mean:+.2f}",
    f"80% CI [{beta_lo:+.2f}, {beta_hi:+.2f}]",
)
c4.metric(
    "Capacity-bound %",
    f"{adopted['capacity_binds'].mean() * 100:.0f}%",
    "of adopted recommendations",
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Posterior + uplift distribution
# ---------------------------------------------------------------------------

a, b = st.columns([1, 1.3])

with a:
    st.markdown("### Where the elasticity number comes from")
    fig = go.Figure()
    fig.add_histogram(
        x=post["beta_samples"], nbinsx=40,
        marker_color="#2E5C8A", opacity=0.85, name="posterior",
    )
    fig.add_vline(x=beta_mean, line_dash="dash", line_color="#1A1F2B",
                  annotation_text=f"β = {beta_mean:.2f}", annotation_position="top right")
    fig.add_vline(x=0, line_dash="dot", line_color="#C04848",
                  annotation_text="0 (no effect)", annotation_position="top left")
    fig.update_layout(
        height=320,
        margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="β (log_price coefficient)",
        yaxis_title="posterior samples",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "After conditioning on bookings-on-books to close the demand→price backdoor, "
        "the global elasticity is firmly negative. |β| < 1 implies inelastic demand: "
        "revenue grows when prices rise, until capacity binds."
    )

with b:
    st.markdown("### Uplift by market group")
    by_market = (adopted.groupby("MarketGroupCode")["uplift_eur"]
                       .sum().reset_index().sort_values("uplift_eur", ascending=True))
    fig = px.bar(by_market, x="uplift_eur", y="MarketGroupCode",
                 orientation="h", color_discrete_sequence=["#2E5C8A"])
    fig.update_layout(
        height=320,
        margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Expected uplift (€)", yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Layer 2 — Decision support
# ---------------------------------------------------------------------------

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## Decision support")
st.caption("Where should the pricing manager look first?  Sorted by uplift, flagged by review need.")

# Flag distribution + scatter
left, right = st.columns([0.9, 1.4])

with left:
    flag_counts = fdf["hitl_flag"].value_counts().reindex(["green", "yellow", "red"]).fillna(0)
    fig = go.Figure()
    fig.add_bar(
        x=flag_counts.index, y=flag_counts.values,
        marker_color=["#4C9A4A", "#E0B341", "#C04848"],
        text=flag_counts.values, textposition="outside",
    )
    fig.update_layout(
        height=300,
        margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="HITL flag", yaxis_title="ROMGID count",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
        - **Green**: small price change, stable history → safe to auto-accept
        - **Yellow**: moderate change → light review
        - **Red**: large change or volatile history → senior manager sign-off
        """
    )

with right:
    fig = px.scatter(
        fdf, x="price_change_pct", y="uplift_eur",
        color="hitl_flag",
        color_discrete_map={"green": "#4C9A4A", "yellow": "#E0B341", "red": "#C04848"},
        hover_data=["CampsiteCode", "MarketGroupCode", "AccoTypeRangeCode",
                    "p_obs_mean", "recommended_price", "TBN"],
        opacity=0.55,
    )
    fig.update_layout(
        height=300, margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Recommended price change (%)",
        yaxis_title="Expected uplift (€)",
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    st.plotly_chart(fig, use_container_width=True)


# Top decisions table
st.markdown("### Top 25 decisions ranked by uplift")
table = (fdf.sort_values("uplift_eur", ascending=False)
            .head(25)
            .loc[:, ["CampsiteCode", "MarketGroupCode", "AccoTypeRangeCode",
                     "WeekStartDate", "p_obs_mean", "recommended_price",
                     "price_change_pct", "TBN", "expected_bookings_capped",
                     "uplift_eur", "hitl_flag"]]
            .rename(columns={
                "p_obs_mean": "current price",
                "recommended_price": "recommended",
                "price_change_pct": "Δ %",
                "TBN": "current bookings",
                "expected_bookings_capped": "expected bookings",
                "uplift_eur": "uplift €",
                "hitl_flag": "flag",
            }))
st.dataframe(
    table.style.format({
        "current price": "€{:.2f}",
        "recommended": "€{:.2f}",
        "Δ %": "{:+.1f}%",
        "expected bookings": "{:.1f}",
        "uplift €": "€{:,.0f}",
    }),
    use_container_width=True,
    hide_index=True,
    height=420,
)


# ---------------------------------------------------------------------------
# Layer 3 — Strategic insight
# ---------------------------------------------------------------------------

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## Strategic insight")

# Heatmap: market × type uplift
pivot = (fdf.groupby(["MarketGroupCode", "AccoTypeRangeCode"])["uplift_eur"]
            .sum().reset_index()
            .pivot(index="MarketGroupCode", columns="AccoTypeRangeCode",
                   values="uplift_eur")
            .fillna(0))

st.markdown("### Where does the uplift come from?")
fig = px.imshow(
    pivot.values,
    labels=dict(x="Accommodation type", y="Market group", color="Uplift €"),
    x=pivot.columns, y=pivot.index,
    color_continuous_scale="Blues",
    aspect="auto",
)
fig.update_layout(
    height=320, margin=dict(t=10, b=10, l=10, r=10),
    plot_bgcolor="white", paper_bgcolor="white",
)
st.plotly_chart(fig, use_container_width=True)


# Scenario analyzer
st.markdown("### Scenario analyzer")
st.caption("What if we apply a uniform price change instead of the model's recommendation?")

scen_col1, scen_col2 = st.columns([1, 2])

with scen_col1:
    delta = st.slider("Uniform price change (%)", -15, 25, 5, step=1)
    beta = beta_mean
    fdf2 = fdf.copy()
    new_price = fdf2["p_obs_mean"] * (1 + delta / 100)
    new_bookings = fdf2["TBN"] * (new_price / fdf2["p_obs_mean"]) ** beta
    new_bookings_capped = np.minimum(new_bookings, fdf2["capacity"])
    scenario_revenue = new_price * new_bookings_capped
    scenario_uplift = (scenario_revenue - fdf2["observed_revenue"]).sum()
    model_uplift = adopted["uplift_eur"].sum()

    st.metric("Scenario uplift", f"€{scenario_uplift:,.0f}")
    st.metric("Model uplift (for comparison)", f"€{model_uplift:,.0f}")
    st.caption(
        f"At Δ = {delta:+d}% the constant-elasticity formula predicts revenue "
        f"to {'rise' if scenario_uplift > 0 else 'fall'}. Beyond ±15% you are "
        "extrapolating outside the support of the training data — flag for senior review."
    )

with scen_col2:
    deltas = np.arange(-15, 26, 1)
    uplifts = []
    for d in deltas:
        np_ = fdf["p_obs_mean"] * (1 + d / 100)
        nb_ = fdf["TBN"] * (np_ / fdf["p_obs_mean"]) ** beta_mean
        nbc_ = np.minimum(nb_, fdf["capacity"])
        rev_ = np_ * nbc_
        uplifts.append((rev_ - fdf["observed_revenue"]).sum())

    fig = go.Figure()
    fig.add_scatter(x=deltas, y=uplifts, mode="lines",
                    line=dict(color="#2E5C8A", width=3))
    fig.add_vline(x=delta, line_dash="dash", line_color="#1A1F2B")
    fig.add_hline(y=0, line_color="#888", line_width=1)
    fig.add_vrect(x0=15, x1=25, fillcolor="#C04848", opacity=0.08, line_width=0,
                  annotation_text="extrapolation zone", annotation_position="top left")
    fig.update_layout(
        height=320, margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Uniform price change (%)",
        yaxis_title="Total expected uplift (€)",
    )
    st.plotly_chart(fig, use_container_width=True)


# Footer
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption(
    "Model: Bambi Negative-Binomial hierarchical regression with random elasticity per "
    "(campsite × accommodation × market). Conditioning on bookings-on-books closes "
    "the demand→price backdoor. Validation: time-based holdout + PSIS-LOO. "
    "Recommendations capped at ±15% of observed price range. "
    "All numbers are model estimates — flag-coded for review."
)
