# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import date

st.set_page_config(page_title="OI Progression", page_icon="📈", layout="wide")

DB_PATH = Path(__file__).parent.parent / "Database"

COMMODITIES = {
    "KC":  ("kc_futures.parquet",  "Coffee (KC)"),
    "CC":  ("cc_futures.parquet",  "Cocoa (CC)"),
    "CT":  ("ct_futures.parquet",  "Cotton (CT)"),
    "SB":  ("sb_futures.parquet",  "Sugar #11 (SB)"),
    "RC":  ("rc_futures.parquet",  "Robusta (RC)"),
    "LCC": ("lcc_futures.parquet", "Liffe Cocoa (LCC)"),
    "LSU": ("lsu_futures.parquet", "Liffe Sugar (LSU)"),
}

MONTH_NAMES = {
    "F": "January", "G": "February", "H": "March",  "J": "April",
    "K": "May",     "M": "June",     "N": "July",   "Q": "August",
    "U": "September","V": "October", "X": "November","Z": "December",
}

COLORS = {
    "band_outer": "rgba(99, 149, 237, 0.10)",
    "band_inner": "rgba(99, 149, 237, 0.28)",
    "avg":        "#4A7FD4",
    "current":    "#E8470A",
    "individual": "rgba(160, 160, 160, 0.45)",
    "grid":       "rgba(0, 0, 0, 0.07)",
    "bg":         "#ffffff",
    "font":       "#1a1a1a",
    "vline":      "rgba(0, 0, 0, 0.18)",
}


@st.cache_data
def load_data(commodity: str) -> pd.DataFrame:
    filename, _ = COMMODITIES[commodity]
    df = pd.read_parquet(DB_PATH / filename)
    df["Date"] = pd.to_datetime(df["Date"])
    df["LTD"]  = pd.to_datetime(df["LTD"])
    df["days_to_expiry"] = (df["LTD"] - df["Date"]).dt.days
    return df[df["open_interest"] > 0].copy()


@st.cache_data
def load_data_with_share(commodity: str) -> pd.DataFrame:
    """Same as load_data but adds oi_share_pct = contract OI / total commodity OI on that date."""
    df = load_data(commodity)
    total_oi = df.groupby("Date")["open_interest"].sum().rename("total_oi")
    df = df.merge(total_oi, on="Date")
    df["oi_share_pct"] = df["open_interest"] / df["total_oi"] * 100
    return df


def compute_share_band(commodity, month, hist_year_range, smooth_window=7):
    """Returns (band, curr_df, current_sym) using oi_share_pct as the metric."""
    df    = load_data_with_share(commodity)
    dm    = df[df["month"] == month].copy()
    today = pd.Timestamp(date.today())

    ltd_by_sym  = dm.groupby("ice_symbol")["LTD"].first()
    active_syms = ltd_by_sym[ltd_by_sym >= today].sort_values().index.tolist()
    hist_syms   = ltd_by_sym[ltd_by_sym <  today].sort_values().index.tolist()

    if not active_syms:
        return None

    hist_df = dm[
        dm["ice_symbol"].isin(hist_syms) &
        dm["year"].between(hist_year_range[0], hist_year_range[1])
    ]

    band = (
        hist_df.groupby("days_to_expiry")["oi_share_pct"]
        .agg(
            hist_min="min",
            hist_max="max",
            hist_mean="mean",
            hist_q25=lambda x: x.quantile(0.25),
            hist_q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
        .sort_values("days_to_expiry")
    )
    for col in ["hist_min", "hist_max", "hist_mean", "hist_q25", "hist_q75"]:
        band[col] = band[col].rolling(smooth_window, center=True, min_periods=1).mean()

    curr_df = dm[dm["ice_symbol"] == active_syms[0]].sort_values("Date").copy()
    return band, curr_df, active_syms[0]


def compute_band_and_current(commodity, month, hist_year_range, smooth_window=7):
    """Returns (band, curr_df, current_sym, active_syms, hist_syms) or None if no active contract."""
    df      = load_data(commodity)
    dm      = df[df["month"] == month].copy()
    today   = pd.Timestamp(date.today())

    ltd_by_sym  = dm.groupby("ice_symbol")["LTD"].first()
    active_syms = ltd_by_sym[ltd_by_sym >= today].sort_values().index.tolist()
    hist_syms   = ltd_by_sym[ltd_by_sym <  today].sort_values().index.tolist()

    if not active_syms:
        return None

    hist_df = dm[
        dm["ice_symbol"].isin(hist_syms) &
        dm["year"].between(hist_year_range[0], hist_year_range[1])
    ]

    band = (
        hist_df.groupby("days_to_expiry")["open_interest"]
        .agg(
            hist_min="min",
            hist_max="max",
            hist_mean="mean",
            hist_q25=lambda x: x.quantile(0.25),
            hist_q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
        .sort_values("days_to_expiry")
    )
    for col in ["hist_min", "hist_max", "hist_mean", "hist_q25", "hist_q75"]:
        band[col] = band[col].rolling(smooth_window, center=True, min_periods=1).mean()

    curr_df = dm[dm["ice_symbol"] == active_syms[0]].sort_values("Date")

    return band, curr_df, active_syms[0], active_syms, hist_syms


def add_oi_traces(fig, band, curr_df, current_sym, oi_fmt, show_individual=False,
                  hist_df=None, row=None, col=None, show_legend=True):
    """Add all OI traces to a figure (works for both single chart and subplots)."""
    kw = dict(row=row, col=col) if row else {}

    # Outer band: min-max
    fig.add_trace(go.Scatter(
        x=band["days_to_expiry"], y=band["hist_max"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip", name="_max",
    ), **kw)
    fig.add_trace(go.Scatter(
        x=band["days_to_expiry"], y=band["hist_min"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=COLORS["band_outer"],
        name="Min-Max Range", showlegend=show_legend,
        hoverinfo="skip", legendgroup="outer_band",
    ), **kw)

    # Inner band: q25-q75
    fig.add_trace(go.Scatter(
        x=band["days_to_expiry"], y=band["hist_q75"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip", name="_q75",
    ), **kw)
    fig.add_trace(go.Scatter(
        x=band["days_to_expiry"], y=band["hist_q25"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=COLORS["band_inner"],
        name="25th-75th Pct", showlegend=show_legend,
        hoverinfo="skip", legendgroup="inner_band",
    ), **kw)

    # Mean line
    fig.add_trace(go.Scatter(
        x=band["days_to_expiry"], y=band["hist_mean"],
        mode="lines", line=dict(color=COLORS["avg"], width=2, dash="dash"),
        name="Historical Mean", showlegend=show_legend,
        hovertemplate=f"DTE: %{{x}}<br>Mean: %{{y:{oi_fmt}}}<extra>Mean</extra>",
        legendgroup="mean",
    ), **kw)

    # Individual years
    if show_individual and hist_df is not None:
        for sym, grp in hist_df.groupby("ice_symbol"):
            grp = grp.sort_values("days_to_expiry")
            fig.add_trace(go.Scatter(
                x=grp["days_to_expiry"], y=grp["open_interest"],
                mode="lines", line=dict(width=0.9, color=COLORS["individual"]),
                name=sym, showlegend=show_legend,
                hovertemplate=f"{sym}  DTE: %{{x}}  OI: %{{y:{oi_fmt}}}<extra></extra>",
            ), **kw)

    # Current contract line
    fig.add_trace(go.Scatter(
        x=curr_df["days_to_expiry"], y=curr_df["open_interest"],
        mode="lines", line=dict(color=COLORS["current"], width=2.5),
        name=current_sym, showlegend=show_legend,
        hovertemplate=f"<b>{current_sym}</b><br>DTE: %{{x}}<br>OI: %{{y:{oi_fmt}}}<extra></extra>",
        legendgroup="current",
    ), **kw)

    # Latest dot
    latest = curr_df.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[int(latest["days_to_expiry"])], y=[latest["open_interest"]],
        mode="markers",
        marker=dict(color=COLORS["current"], size=8, line=dict(color="white", width=1.5)),
        showlegend=False,
        hovertemplate=(
            f"<b>{latest['Date'].strftime('%b %d, %Y')}</b><br>"
            f"DTE: {int(latest['days_to_expiry'])}<br>"
            f"OI: {latest['open_interest']:{oi_fmt}}<extra></extra>"
        ),
    ), **kw)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    st.markdown("---")

    commodity = st.selectbox(
        "Commodity",
        list(COMMODITIES.keys()),
        format_func=lambda x: COMMODITIES[x][1],
    )

    df_sidebar = load_data(commodity)
    available_months = sorted(df_sidebar["month"].unique())
    default_idx = available_months.index("N") if "N" in available_months else 0

    selected_month = st.selectbox(
        "Contract Month",
        available_months,
        index=default_idx,
        format_func=lambda x: f"{MONTH_NAMES.get(x, x)} ({x})",
    )

    df_month   = df_sidebar[df_sidebar["month"] == selected_month].copy()
    today      = pd.Timestamp(date.today())
    ltd_by_sym = df_month.groupby("ice_symbol")["LTD"].first()
    active_syms = ltd_by_sym[ltd_by_sym >= today].sort_values().index.tolist()
    hist_syms   = ltd_by_sym[ltd_by_sym <  today].sort_values().index.tolist()

    if not active_syms:
        st.error("No active contract found.")
        st.stop()

    current_contract = st.selectbox("Current Contract", active_syms)

    st.markdown("---")

    years_all = sorted(df_month[df_month["ice_symbol"].isin(hist_syms)]["year"].unique())
    hist_range = st.slider(
        "Historical Years",
        min_value=int(years_all[0]), max_value=int(years_all[-1]),
        value=(int(years_all[0]), int(years_all[-1])),
    ) if years_all else (0, 0)

    st.markdown("---")

    # Reversed DTE slider: left = far from expiry, right = near expiry (matches chart)
    max_dte = int(df_month["days_to_expiry"].max())
    max_dte_r = (max_dte // 10) * 10
    dte_options_rev = list(range(max_dte_r, -1, -10))  # [1180, 1170, ..., 10, 0]
    dte_sel = st.select_slider(
        "Days to Expiry Range",
        options=dte_options_rev,
        value=(dte_options_rev[0], dte_options_rev[-1]),  # (far, near)
    )
    # dte_sel[0] = left handle (high DTE), dte_sel[1] = right handle (low DTE)

    st.markdown("---")
    show_individual = st.toggle("Show individual years", value=False)
    normalize       = st.toggle("Normalize OI (% of peak)", value=False)


# ── Main data ─────────────────────────────────────────────────────────────────
result = compute_band_and_current(commodity, selected_month, hist_range)
if result is None:
    st.error("No data available.")
    st.stop()

band, curr_df, _, active_syms, hist_syms = result

# Override current_contract from sidebar selection
curr_df = df_month[df_month["ice_symbol"] == current_contract].sort_values("Date").copy()

if normalize:
    hist_df_full = df_month[
        df_month["ice_symbol"].isin(hist_syms) &
        df_month["year"].between(hist_range[0], hist_range[1])
    ].copy()
    for sym, grp in hist_df_full.groupby("ice_symbol"):
        peak = grp["open_interest"].max()
        if peak > 0:
            hist_df_full.loc[grp.index, "open_interest"] = grp["open_interest"] / peak * 100
    curr_peak = curr_df["open_interest"].max()
    if curr_peak > 0:
        curr_df["open_interest"] = curr_df["open_interest"] / curr_peak * 100
    band, curr_df_norm, *_ = compute_band_and_current(commodity, selected_month, hist_range) or (band, curr_df)

oi_fmt  = ".1f" if normalize else ",.0f"
oi_unit = "% of peak" if normalize else "contracts"

latest      = curr_df.iloc[-1]
dte_now     = int(latest["days_to_expiry"])
latest_oi   = latest["open_interest"]
latest_date = latest["Date"].strftime("%b %d, %Y")

closest_idx = (band["days_to_expiry"] - dte_now).abs().idxmin()
band_now    = band.loc[closest_idx]
avg_oi      = band_now["hist_mean"]
pct_vs_avg  = (latest_oi - avg_oi) / avg_oi * 100 if avg_oi > 0 else 0.0


# ── Main chart ────────────────────────────────────────────────────────────────
month_name = MONTH_NAMES.get(selected_month, selected_month)
fig = go.Figure()

hist_df_ind = df_month[df_month["ice_symbol"].isin(hist_syms)].copy() if show_individual else None
add_oi_traces(fig, band, curr_df, current_contract, oi_fmt,
              show_individual=show_individual, hist_df=hist_df_ind)

fig.add_vline(x=dte_now, line=dict(color=COLORS["vline"], width=1, dash="dot"))

fig.update_layout(
    title=dict(
        text=f"<b>{commodity} {month_name}</b>  |  Open Interest Progression",
        font=dict(size=18, color=COLORS["font"]), x=0.01,
    ),
    xaxis=dict(
        title="Days to Expiry",
        range=[dte_sel[0], dte_sel[1]],
        showgrid=True, gridcolor=COLORS["grid"],
        zeroline=False, tickfont=dict(size=12, color=COLORS["font"]),
    ),
    yaxis=dict(
        title=f"Open Interest ({oi_unit})",
        showgrid=True, gridcolor=COLORS["grid"],
        tickformat="," if not normalize else ".1f",
        zeroline=False, tickfont=dict(size=12, color=COLORS["font"]),
    ),
    plot_bgcolor=COLORS["bg"],
    paper_bgcolor=COLORS["bg"],
    font=dict(color=COLORS["font"], family="Inter, sans-serif"),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.01,
        xanchor="left", x=0,
        bgcolor="rgba(255,255,255,0)", font=dict(size=11),
    ),
    hovermode="x unified",
    height=540,
    margin=dict(l=70, r=30, t=80, b=60),
)


# ── Page ──────────────────────────────────────────────────────────────────────
st.markdown("## OI Progression")
st.markdown("---")

st.markdown("""
<style>
[data-testid="stMetricLabel"] { font-size: 0.70rem !important; color: #888; }
[data-testid="stMetricValue"] { font-size: 1.10rem !important; font-weight: 600; }
[data-testid="stMetricDelta"] { font-size: 0.70rem !important; }
</style>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Contract",       current_contract)
c2.metric("Current OI",     f"{latest_oi:,.0f}" if not normalize else f"{latest_oi:.1f}%")
c3.metric("As of",          latest_date)
c4.metric("Days to Expiry", dte_now)
c5.metric(
    "vs Hist Mean",
    f"{avg_oi:,.0f}" if not normalize else f"{avg_oi:.1f}%",
    delta=f"{pct_vs_avg:+.1f}%",
    delta_color="normal",
)

st.plotly_chart(fig, use_container_width=True)


# ── 2x2 Multi-contract view: 4 nearest active contracts for selected commodity ─
st.markdown("---")
st.markdown(f"### {COMMODITIES[commodity][1]} — 4 Active Contracts")

# Find the 4 nearest active contracts for the selected commodity
df_all_months = load_data(commodity)
ltd_all = (
    df_all_months.groupby("ice_symbol")[["LTD", "month"]]
    .first()
    .reset_index()
)
active_all = ltd_all[ltd_all["LTD"] >= today].sort_values("LTD").head(4)

if len(active_all) < 4:
    st.info(f"Only {len(active_all)} active contracts found for {commodity}.")

quad_contracts = list(active_all["ice_symbol"])
quad_months    = list(active_all["month"])

subplot_titles = [
    f"{sym}  ({MONTH_NAMES.get(m, m)})" for sym, m in zip(quad_contracts, quad_months)
]

fig4 = make_subplots(
    rows=2, cols=2,
    subplot_titles=subplot_titles,
    horizontal_spacing=0.08,
    vertical_spacing=0.14,
)

for idx, (sym_q, m_q) in enumerate(zip(quad_contracts, quad_months)):
    r  = idx // 2 + 1
    cl = idx % 2 + 1

    res = compute_band_and_current(commodity, m_q, hist_range)
    if res is None:
        continue
    b_q, _, _, active_q, _ = res

    # Use the specific active contract for this subplot
    curr_q = df_all_months[df_all_months["ice_symbol"] == sym_q].sort_values("Date").copy()

    add_oi_traces(
        fig4, b_q, curr_q, sym_q,
        oi_fmt=",.0f",
        show_individual=False,
        row=r, col=cl,
        show_legend=False,
    )

fig4.update_layout(
    height=720,
    plot_bgcolor=COLORS["bg"],
    paper_bgcolor=COLORS["bg"],
    font=dict(color=COLORS["font"], family="Inter, sans-serif"),
    showlegend=False,
    margin=dict(l=50, r=30, t=60, b=50),
)
for i in range(1, 5):
    fig4.update_xaxes(
        autorange="reversed", showgrid=True, gridcolor=COLORS["grid"],
        tickfont=dict(size=10), zeroline=False,
        row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1,
    )
    fig4.update_yaxes(
        showgrid=True, gridcolor=COLORS["grid"], tickformat=",",
        tickfont=dict(size=10), zeroline=False,
        row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1,
    )

st.plotly_chart(fig4, use_container_width=True)

# Data tables
with st.expander("Current Contract Data", expanded=False):
    tbl = curr_df[["Date", "days_to_expiry", "open_interest", "volume", "settlement"]].copy()
    tbl = tbl.sort_values("Date", ascending=False)
    tbl.columns = ["Date", "DTE", "Open Interest", "Volume", "Settlement"]
    tbl["Date"] = tbl["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(tbl, use_container_width=True, hide_index=True)


# ── OI Share chart ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### {current_contract} — Share of Total {commodity} Market OI (%)")
st.caption("Contract OI divided by sum of all active contracts' OI on that date.")

share_result = compute_share_band(commodity, selected_month, hist_range)

if share_result is not None:
    s_band, s_curr, s_sym = share_result

    # Override with sidebar-selected contract
    df_share_full = load_data_with_share(commodity)
    s_curr = df_share_full[df_share_full["ice_symbol"] == current_contract].sort_values("Date").copy()

    s_latest    = s_curr.iloc[-1]
    s_dte_now   = int(s_latest["days_to_expiry"])
    s_share_now = s_latest["oi_share_pct"]
    s_date_str  = s_latest["Date"].strftime("%b %d, %Y")

    s_closest   = (s_band["days_to_expiry"] - s_dte_now).abs().idxmin()
    s_band_now  = s_band.loc[s_closest]
    s_avg       = s_band_now["hist_mean"]
    s_pct_diff  = (s_share_now - s_avg)

    # KPI row
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Contract",        current_contract)
    sc2.metric("Current Share",   f"{s_share_now:.1f}%")
    sc3.metric("As of",           s_date_str)
    sc4.metric("vs Hist Mean",    f"{s_avg:.1f}%", delta=f"{s_pct_diff:+.1f}pp")

    # Build chart
    fig_share = go.Figure()

    # Outer band min-max
    fig_share.add_trace(go.Scatter(
        x=s_band["days_to_expiry"], y=s_band["hist_max"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_share.add_trace(go.Scatter(
        x=s_band["days_to_expiry"], y=s_band["hist_min"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(52, 168, 83, 0.10)",
        name="Min-Max Range", hoverinfo="skip",
    ))

    # Inner band q25-q75
    fig_share.add_trace(go.Scatter(
        x=s_band["days_to_expiry"], y=s_band["hist_q75"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_share.add_trace(go.Scatter(
        x=s_band["days_to_expiry"], y=s_band["hist_q25"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(52, 168, 83, 0.25)",
        name="25th-75th Pct", hoverinfo="skip",
    ))

    # Mean line
    fig_share.add_trace(go.Scatter(
        x=s_band["days_to_expiry"], y=s_band["hist_mean"],
        mode="lines", line=dict(color="#34A853", width=2, dash="dash"),
        name="Historical Mean",
        hovertemplate="DTE: %{x}<br>Mean Share: %{y:.1f}%<extra>Mean</extra>",
    ))

    # Current contract line
    fig_share.add_trace(go.Scatter(
        x=s_curr["days_to_expiry"], y=s_curr["oi_share_pct"],
        mode="lines", line=dict(color="#E8470A", width=2.5),
        name=current_contract,
        hovertemplate=f"<b>{current_contract}</b><br>DTE: %{{x}}<br>Share: %{{y:.1f}}%<extra></extra>",
    ))

    # Latest dot
    fig_share.add_trace(go.Scatter(
        x=[s_dte_now], y=[s_share_now],
        mode="markers",
        marker=dict(color="#E8470A", size=8, line=dict(color="white", width=1.5)),
        showlegend=False,
        hovertemplate=f"<b>{s_date_str}</b><br>DTE: {s_dte_now}<br>Share: {s_share_now:.1f}%<extra></extra>",
    ))

    # DTE reference line
    fig_share.add_vline(x=s_dte_now, line=dict(color=COLORS["vline"], width=1, dash="dot"))

    fig_share.update_layout(
        xaxis=dict(
            title="Days to Expiry",
            range=[dte_sel[0], dte_sel[1]],
            showgrid=True, gridcolor=COLORS["grid"],
            zeroline=False, tickfont=dict(size=12, color=COLORS["font"]),
        ),
        yaxis=dict(
            title="Share of Total Market OI (%)",
            showgrid=True, gridcolor=COLORS["grid"],
            tickformat=".1f", ticksuffix="%",
            zeroline=False, tickfont=dict(size=12, color=COLORS["font"]),
        ),
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["font"], family="Inter, sans-serif"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            bgcolor="rgba(255,255,255,0)", font=dict(size=11),
        ),
        hovermode="x unified",
        height=500,
        margin=dict(l=70, r=30, t=40, b=60),
    )

    st.plotly_chart(fig_share, use_container_width=True)
else:
    st.info("No historical data available for OI share chart.")
