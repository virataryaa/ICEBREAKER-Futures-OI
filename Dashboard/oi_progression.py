# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import date

st.set_page_config(page_title="Futures Dashboard", page_icon="📈", layout="wide")

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

C = {
    # OI charts
    "oi_outer":  "rgba(99, 149, 237, 0.10)",
    "oi_inner":  "rgba(99, 149, 237, 0.28)",
    "oi_avg":    "#4A7FD4",
    # OI share
    "sh_outer":  "rgba(52, 168, 83, 0.10)",
    "sh_inner":  "rgba(52, 168, 83, 0.25)",
    "sh_avg":    "#34A853",
    # Vol/OI ratio
    "vr_outer":  "rgba(20, 184, 166, 0.10)",
    "vr_inner":  "rgba(20, 184, 166, 0.28)",
    "vr_avg":    "#0D9488",
    # Vol market share
    "vs_outer":  "rgba(139, 92, 246, 0.10)",
    "vs_inner":  "rgba(139, 92, 246, 0.28)",
    "vs_avg":    "#7C3AED",
    # Rolling volume
    "rv_outer":  "rgba(245, 158, 11, 0.10)",
    "rv_inner":  "rgba(245, 158, 11, 0.28)",
    "rv_avg":    "#D97706",
    # Common
    "current":   "#E8470A",
    "individual":"rgba(160,160,160,0.4)",
    "grid":      "rgba(0,0,0,0.07)",
    "bg":        "#ffffff",
    "font":      "#1a1a1a",
    "vline":     "rgba(0,0,0,0.18)",
}


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(commodity: str) -> pd.DataFrame:
    filename, _ = COMMODITIES[commodity]
    df = pd.read_parquet(DB_PATH / filename)
    df["Date"] = pd.to_datetime(df["Date"])
    df["LTD"]  = pd.to_datetime(df["LTD"])
    df["days_to_expiry"] = (df["LTD"] - df["Date"]).dt.days
    return df[df["open_interest"] > 0].copy()


@st.cache_data
def load_enriched(commodity: str) -> pd.DataFrame:
    """Adds oi_share_pct, vol_share_pct, vol_oi_ratio to every row."""
    df = load_data(commodity)
    tot_oi  = df.groupby("Date")["open_interest"].sum().rename("total_oi")
    tot_vol = df.groupby("Date")["volume"].sum().rename("total_vol")
    df = df.merge(tot_oi, on="Date").merge(tot_vol, on="Date")
    df["oi_share_pct"] = df["open_interest"] / df["total_oi"]  * 100
    df["vol_share_pct"]= df["volume"]        / df["total_vol"] * 100
    df["vol_oi_ratio"] = df["volume"]        / df["open_interest"]
    return df


# ── Band computation ──────────────────────────────────────────────────────────
def _split_contracts(dm):
    today = pd.Timestamp(date.today())
    ltd   = dm.groupby("ice_symbol")["LTD"].first()
    return (ltd[ltd >= today].sort_values().index.tolist(),
            ltd[ltd <  today].sort_values().index.tolist())


def compute_band(commodity, month, hist_year_range, metric_col,
                 roll_n=None, smooth_window=7, use_enriched=False):
    """
    Generic band + current-contract computation for any metric.
    roll_n: if set, compute rolling(roll_n).mean() on 'volume' first (by Date order).
    Returns (band, curr_df, active_syms, hist_syms) or None.
    """
    df = load_enriched(commodity) if use_enriched else load_data(commodity)
    dm = df[df["month"] == month].copy()

    active_syms, hist_syms = _split_contracts(dm)
    if not active_syms:
        return None

    if roll_n:
        pieces = []
        for sym, grp in dm.groupby("ice_symbol"):
            g = grp.sort_values("Date").copy()
            g["_metric"] = g["volume"].rolling(roll_n, min_periods=1).mean()
            pieces.append(g)
        dm = pd.concat(pieces)
        metric_col = "_metric"

    hist_df = dm[
        dm["ice_symbol"].isin(hist_syms) &
        dm["year"].between(hist_year_range[0], hist_year_range[1])
    ]

    band = (
        hist_df.groupby("days_to_expiry")[metric_col]
        .agg(hist_min="min", hist_max="max", hist_mean="mean",
             hist_q25=lambda x: x.quantile(0.25),
             hist_q75=lambda x: x.quantile(0.75))
        .reset_index().sort_values("days_to_expiry")
    )
    for c in ["hist_min","hist_max","hist_mean","hist_q25","hist_q75"]:
        band[c] = band[c].rolling(smooth_window, center=True, min_periods=1).mean()

    curr_df = dm[dm["ice_symbol"] == active_syms[0]].sort_values("Date").copy()
    return band, curr_df, active_syms, hist_syms


# ── Generic chart builder ─────────────────────────────────────────────────────
def build_chart(band, curr_df, metric_col, current_sym,
                title, y_title, y_fmt, y_suffix,
                outer_color, inner_color, avg_color,
                dte_range, dte_now,
                show_individual=False, hist_df=None, ind_metric=None,
                height=500):
    fig = go.Figure()

    # Outer band
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_max"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_min"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=outer_color,
        name="Min-Max Range", hoverinfo="skip"))

    # Inner band q25-q75
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_q75"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_q25"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=inner_color,
        name="25th-75th Pct", hoverinfo="skip"))

    # Mean line
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_mean"],
        mode="lines", line=dict(color=avg_color, width=2, dash="dash"),
        name="Historical Mean",
        hovertemplate=f"DTE: %{{x}}<br>Mean: %{{y:{y_fmt}}}{y_suffix}<extra>Mean</extra>"))

    # Individual years
    if show_individual and hist_df is not None and ind_metric:
        for sym, grp in hist_df.groupby("ice_symbol"):
            grp = grp.sort_values("days_to_expiry")
            fig.add_trace(go.Scatter(x=grp["days_to_expiry"], y=grp[ind_metric],
                mode="lines", line=dict(width=0.9, color=C["individual"]),
                name=sym, showlegend=True,
                hovertemplate=f"{sym} DTE:%{{x}} %{{y:{y_fmt}}}<extra></extra>"))

    # Current line
    fig.add_trace(go.Scatter(x=curr_df["days_to_expiry"], y=curr_df[metric_col],
        mode="lines", line=dict(color=C["current"], width=2.5),
        name=current_sym,
        hovertemplate=f"<b>{current_sym}</b><br>DTE: %{{x}}<br>%{{y:{y_fmt}}}{y_suffix}<extra></extra>"))

    # Latest dot
    latest = curr_df.iloc[-1]
    lat_val = latest[metric_col]
    lat_dte = int(latest["days_to_expiry"])
    lat_dt  = latest["Date"].strftime("%b %d, %Y")
    fig.add_trace(go.Scatter(x=[lat_dte], y=[lat_val],
        mode="markers",
        marker=dict(color=C["current"], size=8, line=dict(color="white", width=1.5)),
        showlegend=False,
        hovertemplate=f"<b>{lat_dt}</b><br>DTE: {lat_dte}<br>{lat_val:{y_fmt}}{y_suffix}<extra></extra>"))

    fig.add_vline(x=dte_now, line=dict(color=C["vline"], width=1, dash="dot"))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=C["font"]), x=0.01),
        xaxis=dict(title="Days to Expiry", range=[dte_range[0], dte_range[1]],
                   showgrid=True, gridcolor=C["grid"], zeroline=False,
                   tickfont=dict(size=11, color=C["font"])),
        yaxis=dict(title=y_title, showgrid=True, gridcolor=C["grid"],
                   zeroline=False, tickformat=y_fmt.replace(",",",.0f").replace(".1f",".1f"),
                   ticksuffix=y_suffix, tickfont=dict(size=11, color=C["font"])),
        plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
        font=dict(color=C["font"], family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        hovermode="x unified", height=height,
        margin=dict(l=70, r=30, t=60, b=55),
    )
    return fig


def kpi_row(vals: list):
    """vals = list of (label, value, delta, delta_color) — delta/delta_color optional."""
    cols = st.columns(len(vals))
    for col, item in zip(cols, vals):
        label, value = item[0], item[1]
        delta        = item[2] if len(item) > 2 else None
        dc           = item[3] if len(item) > 3 else "normal"
        if delta is not None:
            col.metric(label, value, delta=delta, delta_color=dc)
        else:
            col.metric(label, value)


# ── Subplot trace helper (for 2x2 grid) ──────────────────────────────────────
def add_oi_traces(fig, band, curr_df, current_sym, oi_fmt,
                  show_individual=False, hist_df=None, row=None, col=None, show_legend=True):
    kw = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_max"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), **kw)
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_min"],
        mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C["oi_outer"],
        name="Min-Max", showlegend=show_legend, hoverinfo="skip", legendgroup="outer"), **kw)
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_q75"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), **kw)
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_q25"],
        mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C["oi_inner"],
        name="25-75 Pct", showlegend=show_legend, hoverinfo="skip", legendgroup="inner"), **kw)
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_mean"],
        mode="lines", line=dict(color=C["oi_avg"], width=1.5, dash="dash"),
        name="Mean", showlegend=show_legend, legendgroup="mean",
        hovertemplate=f"DTE: %{{x}}<br>Mean: %{{y:{oi_fmt}}}<extra>Mean</extra>"), **kw)
    fig.add_trace(go.Scatter(x=curr_df["days_to_expiry"], y=curr_df["open_interest"],
        mode="lines", line=dict(color=C["current"], width=2),
        name=current_sym, showlegend=show_legend, legendgroup="curr",
        hovertemplate=f"<b>{current_sym}</b><br>DTE:%{{x}}<br>OI:%{{y:{oi_fmt}}}<extra></extra>"), **kw)
    latest = curr_df.iloc[-1]
    fig.add_trace(go.Scatter(x=[int(latest["days_to_expiry"])], y=[latest["open_interest"]],
        mode="markers", marker=dict(color=C["current"], size=7, line=dict(color="white", width=1.5)),
        showlegend=False,
        hovertemplate=f"<b>{latest['Date'].strftime('%b %d, %Y')}</b><br>OI:{latest['open_interest']:{oi_fmt}}<extra></extra>"), **kw)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    st.markdown("---")

    commodity = st.selectbox("Commodity", list(COMMODITIES.keys()),
                             format_func=lambda x: COMMODITIES[x][1])

    df_sidebar      = load_data(commodity)
    avail_months    = sorted(df_sidebar["month"].unique())
    default_idx     = avail_months.index("N") if "N" in avail_months else 0
    selected_month  = st.selectbox("Contract Month", avail_months, index=default_idx,
                                   format_func=lambda x: f"{MONTH_NAMES.get(x,x)} ({x})")

    df_month    = df_sidebar[df_sidebar["month"] == selected_month].copy()
    today       = pd.Timestamp(date.today())
    active_syms, hist_syms = _split_contracts(df_month)

    if not active_syms:
        st.error("No active contract found.")
        st.stop()

    current_contract = st.selectbox("Current Contract", active_syms)
    st.markdown("---")

    years_all  = sorted(df_month[df_month["ice_symbol"].isin(hist_syms)]["year"].unique())
    hist_range = st.slider("Historical Years",
                           int(years_all[0]), int(years_all[-1]),
                           (int(years_all[0]), int(years_all[-1]))) if years_all else (0,0)
    st.markdown("---")

    max_dte      = int(df_month["days_to_expiry"].max())
    max_dte_r    = (max_dte // 10) * 10
    dte_opts_rev = list(range(max_dte_r, -1, -10))
    dte_sel      = st.select_slider("Days to Expiry Range", options=dte_opts_rev,
                                    value=(dte_opts_rev[0], dte_opts_rev[-1]))
    dte_range    = [dte_sel[0], dte_sel[1]]   # [high DTE, low DTE] — chart is reversed

    st.markdown("---")
    show_individual = st.toggle("Show individual years", value=False)
    normalize       = st.toggle("Normalize OI (% of peak)", value=False)


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricLabel"] { font-size:0.70rem !important; color:#888; }
[data-testid="stMetricValue"] { font-size:1.10rem !important; font-weight:600; }
[data-testid="stMetricDelta"] { font-size:0.70rem !important; }
</style>""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_oi, tab_vol = st.tabs(["OI Progression", "Volume"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OI PROGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_oi:
    res = compute_band(commodity, selected_month, hist_range, "open_interest")
    if res is None:
        st.error("No data available.")
        st.stop()

    band, curr_df, _, _ = res
    curr_df = df_month[df_month["ice_symbol"] == current_contract].sort_values("Date").copy()

    if normalize:
        for sym, grp in df_month[df_month["ice_symbol"].isin(hist_syms)].groupby("ice_symbol"):
            peak = grp["open_interest"].max()
            if peak > 0:
                df_month.loc[grp.index, "open_interest"] = grp["open_interest"] / peak * 100
        curr_peak = curr_df["open_interest"].max()
        if curr_peak > 0:
            curr_df["open_interest"] = curr_df["open_interest"] / curr_peak * 100
        res2 = compute_band(commodity, selected_month, hist_range, "open_interest")
        if res2: band = res2[0]

    oi_fmt  = ".1f" if normalize else ",.0f"
    oi_unit = "% of peak" if normalize else "contracts"

    latest     = curr_df.iloc[-1]
    dte_now    = int(latest["days_to_expiry"])
    latest_oi  = latest["open_interest"]
    lat_date   = latest["Date"].strftime("%b %d, %Y")

    closest    = (band["days_to_expiry"] - dte_now).abs().idxmin()
    avg_oi     = band.loc[closest, "hist_mean"]
    pct_vs_avg = (latest_oi - avg_oi) / avg_oi * 100 if avg_oi > 0 else 0.0

    month_name = MONTH_NAMES.get(selected_month, selected_month)

    kpi_row([
        ("Contract",        current_contract),
        ("Current OI",      f"{latest_oi:,.0f}" if not normalize else f"{latest_oi:.1f}%"),
        ("As of",           lat_date),
        ("Days to Expiry",  str(dte_now)),
        ("vs Hist Mean",    f"{avg_oi:,.0f}" if not normalize else f"{avg_oi:.1f}%",
                            f"{pct_vs_avg:+.1f}%"),
    ])

    hist_df_ind = df_month[df_month["ice_symbol"].isin(hist_syms)].copy() if show_individual else None
    fig_oi = build_chart(
        band, curr_df, "open_interest", current_contract,
        title=f"<b>{commodity} {month_name}</b>  |  Open Interest Progression",
        y_title=f"Open Interest ({oi_unit})",
        y_fmt=",.0f" if not normalize else ".1f", y_suffix="",
        outer_color=C["oi_outer"], inner_color=C["oi_inner"], avg_color=C["oi_avg"],
        dte_range=dte_range, dte_now=dte_now,
        show_individual=show_individual, hist_df=hist_df_ind, ind_metric="open_interest",
        height=540,
    )
    st.plotly_chart(fig_oi, use_container_width=True)

    # ── 2x2 Active contracts ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### {COMMODITIES[commodity][1]} — 4 Active Contracts")

    df_all     = load_data(commodity)
    ltd_all    = df_all.groupby("ice_symbol")[["LTD","month"]].first().reset_index()
    active_all = ltd_all[ltd_all["LTD"] >= today].sort_values("LTD").head(4)
    quad_syms  = list(active_all["ice_symbol"])
    quad_months= list(active_all["month"])

    fig4 = make_subplots(rows=2, cols=2,
        subplot_titles=[f"{s}  ({MONTH_NAMES.get(m,m)})" for s,m in zip(quad_syms,quad_months)],
        horizontal_spacing=0.08, vertical_spacing=0.14)

    for idx, (sym_q, m_q) in enumerate(zip(quad_syms, quad_months)):
        r, cl = idx//2+1, idx%2+1
        res_q = compute_band(commodity, m_q, hist_range, "open_interest")
        if res_q is None: continue
        b_q = res_q[0]
        c_q = df_all[df_all["ice_symbol"] == sym_q].sort_values("Date").copy()
        add_oi_traces(fig4, b_q, c_q, sym_q, ",.0f", row=r, col=cl, show_legend=False)

    fig4.update_layout(height=720, plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
                       font=dict(color=C["font"], family="Inter, sans-serif"),
                       showlegend=False, margin=dict(l=50,r=30,t=60,b=50))
    for i in range(1,5):
        fig4.update_xaxes(autorange="reversed", showgrid=True, gridcolor=C["grid"],
                          tickfont=dict(size=10), zeroline=False,
                          row=(i-1)//2+1, col=(i-1)%2+1)
        fig4.update_yaxes(showgrid=True, gridcolor=C["grid"], tickformat=",",
                          tickfont=dict(size=10), zeroline=False,
                          row=(i-1)//2+1, col=(i-1)%2+1)
    st.plotly_chart(fig4, use_container_width=True)

    # ── OI Market Share ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### {current_contract} — Share of Total {commodity} Market OI (%)")
    st.caption("Contract OI / sum of all active contracts OI on that date.")

    res_sh = compute_band(commodity, selected_month, hist_range, "oi_share_pct", use_enriched=True)
    if res_sh:
        b_sh, _, _, _ = res_sh
        df_enr  = load_enriched(commodity)
        c_sh    = df_enr[df_enr["ice_symbol"] == current_contract].sort_values("Date").copy()
        lat_sh  = c_sh.iloc[-1]
        dte_sh  = int(lat_sh["days_to_expiry"])
        val_sh  = lat_sh["oi_share_pct"]
        idx_sh  = (b_sh["days_to_expiry"] - dte_sh).abs().idxmin()
        avg_sh  = b_sh.loc[idx_sh, "hist_mean"]

        kpi_row([
            ("Contract",      current_contract),
            ("Current Share", f"{val_sh:.1f}%"),
            ("As of",         lat_sh["Date"].strftime("%b %d, %Y")),
            ("vs Hist Mean",  f"{avg_sh:.1f}%", f"{val_sh-avg_sh:+.1f}pp"),
        ])

        fig_sh = build_chart(b_sh, c_sh, "oi_share_pct", current_contract,
            title=f"<b>{commodity} {month_name}</b>  |  OI Market Share",
            y_title="Share of Total Market OI", y_fmt=".1f", y_suffix="%",
            outer_color=C["sh_outer"], inner_color=C["sh_inner"], avg_color=C["sh_avg"],
            dte_range=dte_range, dte_now=dte_sh, height=480)
        st.plotly_chart(fig_sh, use_container_width=True)

    with st.expander("Current Contract Data", expanded=False):
        tbl = curr_df[["Date","days_to_expiry","open_interest","volume","settlement"]].copy()
        tbl = tbl.sort_values("Date", ascending=False)
        tbl.columns = ["Date","DTE","Open Interest","Volume","Settlement"]
        tbl["Date"] = tbl["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(tbl, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VOLUME
# ═══════════════════════════════════════════════════════════════════════════════
with tab_vol:
    st.markdown(f"### {current_contract}  |  Volume Analysis")
    st.markdown("---")

    # Rolling window selector
    roll_n = st.slider("Rolling Window (days)", min_value=1, max_value=30,
                       value=10, step=1,
                       help="Applied to daily volume before plotting progression")

    month_name = MONTH_NAMES.get(selected_month, selected_month)
    df_enr     = load_enriched(commodity)
    df_enr_m   = df_enr[df_enr["month"] == selected_month].copy()

    # ── Chart 1: Volume / OI Ratio ────────────────────────────────────────────
    st.markdown("#### Volume / OI Ratio")
    st.caption("Daily volume divided by open interest — measures turnover rate / speculative activity.")

    res_vr = compute_band(commodity, selected_month, hist_range, "vol_oi_ratio", use_enriched=True)
    if res_vr:
        b_vr, _, _, _ = res_vr
        c_vr  = df_enr[df_enr["ice_symbol"] == current_contract].sort_values("Date").copy()
        lat   = c_vr.iloc[-1]
        v_now = lat["vol_oi_ratio"]
        d_now = int(lat["days_to_expiry"])
        idx_  = (b_vr["days_to_expiry"] - d_now).abs().idxmin()
        avg_  = b_vr.loc[idx_, "hist_mean"]

        kpi_row([
            ("Contract",     current_contract),
            ("Vol/OI Ratio", f"{v_now:.2f}x"),
            ("As of",        lat["Date"].strftime("%b %d, %Y")),
            ("vs Hist Mean", f"{avg_:.2f}x", f"{v_now-avg_:+.2f}x"),
        ])

        fig_vr = build_chart(b_vr, c_vr, "vol_oi_ratio", current_contract,
            title=f"<b>{commodity} {month_name}</b>  |  Volume / OI Ratio",
            y_title="Vol / OI Ratio", y_fmt=".2f", y_suffix="x",
            outer_color=C["vr_outer"], inner_color=C["vr_inner"], avg_color=C["vr_avg"],
            dte_range=dte_range, dte_now=d_now, height=460)
        st.plotly_chart(fig_vr, use_container_width=True)

    st.markdown("---")

    # ── Chart 2: Volume Market Share ──────────────────────────────────────────
    st.markdown("#### Volume Market Share (%)")
    st.caption("Contract daily volume as % of total commodity volume on that date.")

    res_vs = compute_band(commodity, selected_month, hist_range, "vol_share_pct", use_enriched=True)
    if res_vs:
        b_vs, _, _, _ = res_vs
        c_vs  = df_enr[df_enr["ice_symbol"] == current_contract].sort_values("Date").copy()
        lat   = c_vs.iloc[-1]
        v_now = lat["vol_share_pct"]
        d_now = int(lat["days_to_expiry"])
        idx_  = (b_vs["days_to_expiry"] - d_now).abs().idxmin()
        avg_  = b_vs.loc[idx_, "hist_mean"]

        kpi_row([
            ("Contract",       current_contract),
            ("Vol Share",      f"{v_now:.1f}%"),
            ("As of",          lat["Date"].strftime("%b %d, %Y")),
            ("vs Hist Mean",   f"{avg_:.1f}%", f"{v_now-avg_:+.1f}pp"),
        ])

        fig_vs = build_chart(b_vs, c_vs, "vol_share_pct", current_contract,
            title=f"<b>{commodity} {month_name}</b>  |  Volume Market Share",
            y_title="Share of Total Volume", y_fmt=".1f", y_suffix="%",
            outer_color=C["vs_outer"], inner_color=C["vs_inner"], avg_color=C["vs_avg"],
            dte_range=dte_range, dte_now=d_now, height=460)
        st.plotly_chart(fig_vs, use_container_width=True)

    st.markdown("---")

    # ── Chart 3: Rolling N-day Volume ─────────────────────────────────────────
    st.markdown(f"#### Rolling {roll_n}-Day Average Volume")
    st.caption(f"{roll_n}-day rolling mean of daily volume, aligned by days to expiry.")

    res_rv = compute_band(commodity, selected_month, hist_range,
                          metric_col="volume", roll_n=roll_n)
    if res_rv:
        b_rv, c_rv_base, _, _ = res_rv

        # Compute rolling vol for current contract
        c_rv = df_enr[df_enr["ice_symbol"] == current_contract].sort_values("Date").copy()
        c_rv["_metric"] = c_rv["volume"].rolling(roll_n, min_periods=1).mean()

        lat   = c_rv.iloc[-1]
        v_now = lat["_metric"]
        d_now = int(lat["days_to_expiry"])
        idx_  = (b_rv["days_to_expiry"] - d_now).abs().idxmin()
        avg_  = b_rv.loc[idx_, "hist_mean"]

        kpi_row([
            ("Contract",          current_contract),
            (f"{roll_n}d Avg Vol", f"{v_now:,.0f}"),
            ("As of",             lat["Date"].strftime("%b %d, %Y")),
            ("vs Hist Mean",      f"{avg_:,.0f}", f"{(v_now-avg_)/avg_*100:+.1f}%"),
        ])

        fig_rv = build_chart(b_rv, c_rv, "_metric", current_contract,
            title=f"<b>{commodity} {month_name}</b>  |  Rolling {roll_n}-Day Volume",
            y_title=f"{roll_n}-Day Avg Daily Volume (contracts)",
            y_fmt=",.0f", y_suffix="",
            outer_color=C["rv_outer"], inner_color=C["rv_inner"], avg_color=C["rv_avg"],
            dte_range=dte_range, dte_now=d_now, height=460)
        st.plotly_chart(fig_rv, use_container_width=True)
