# dashboard_public.py
"""
Public Streamlit dashboard:
- Upload trades.csv
- Auto-download last 6 months price history via yfinance
- Display portfolio overview, backtest (6 months), charts, efficient frontier (CML),
  tangent allocation, manual rebalance, ad-hoc allocation, Monte Carlo, target SIP calc.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import plotly.graph_objects as go
from scipy.optimize import minimize
import io
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="ETF SIP Dashboard (Public)", layout="wide")
st.title("📊 ETF SIP Dashboard — Public (auto yfinance, 6-month backtest)")

# ----------------------------
# Helper functions
# ----------------------------
def safe_read_trades(uploaded):
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, parse_dates=["date"], dayfirst=True)
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
            return df
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            return pd.DataFrame()
    # fallback
    try:
        df = pd.read_csv("history/trades.csv", parse_dates=["date"], dayfirst=True)
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        return df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading history/trades.csv: {e}")
        return pd.DataFrame()

def ensure_trade_cols(df):
    required = {"date","etf","price","units","amount","type"}
    return required.issubset(set(df.columns))

def ticker_candidates(t):
    # Try Indian NSE suffix first (common), then raw ticker
    t = str(t).strip()
    if t.upper().endswith(".NS"):
        return [t]
    return [t + ".NS", t]

def download_prices_for(etfs, period="6mo"):
    """
    etfs: list of tickers (symbols as in trades.csv). We'll try TICKER.NS then TICKER.
    Returns DataFrame indexed by date with columns equal to normalized tickers (no .NS).
    """
    clean_map = {}   # maps yf_col -> original symbol key
    results = []
    for etf in etfs:
        tried = ticker_candidates(etf)
        df = None
        for cand in tried:
            try:
                tmp = yf.download(cand, period=period, interval="1d", progress=False, auto_adjust=False)
                # tmp may be empty or contain multiple columns
                if tmp.shape[0] == 0:
                    continue
                if "Close" not in tmp.columns:
                    continue
                series = tmp["Close"].rename(etf)
                series.index = pd.to_datetime(series.index).date
                df = series
                break
            except Exception:
                continue
        if df is None:
            st.warning(f"Price data not found for {etf} (tried {tried}). Using NaNs.")
            df = pd.Series(dtype=float, name=etf)
        results.append(df)
    if not results:
        return pd.DataFrame()
    df_all = pd.concat(results, axis=1).sort_index()
    df_all.index = pd.to_datetime(df_all.index)
    return df_all

def compute_xirr(cashflows):
    """
    cashflows: list of (date, amount) where negative = outflow (invested), positive = current value
    returns annualized XIRR or np.nan
    """
    try:
        from scipy.optimize import newton
    except Exception:
        return float("nan")
    try:
        cashflows = sorted([(pd.to_datetime(d), v) for d,v in cashflows], key=lambda x: x[0])
        dates, vals = zip(*cashflows)
        t0 = dates[0]
        times = np.array([(d - t0).days / 365.0 for d in dates])
        def npv(r):
            return np.sum([v / ((1 + r) ** t) for v,t in zip(vals, times)])
        # initial guess
        r = newton(npv, 0.1, tol=1e-6, maxiter=100)
        return r
    except Exception:
        return float("nan")

def annualized_return_from_monthly_rate(monthly_rate, months):
    if months<=0: return 0.0
    return (1+monthly_rate)**12 - 1

def optimize_weights(returns_df, risk_free_rate=0.065, min_return=None,
                     min_weight=0.0, max_weight=1.0):
    """
    Constrained max-Sharpe optimizer like earlier but with min/max weights.
    returns Series indexed by returns_df.columns
    """
    returns_df = returns_df.dropna(axis=1, how='all')
    if returns_df.shape[1] == 0:
        return pd.Series(dtype=float)

    mu = returns_df.mean() * 252
    cov = returns_df.cov() * 252
    n = len(mu)

    def neg_sharpe(w):
        w = np.array(w)
        port_ret = np.dot(w, mu)
        vol = np.sqrt(w @ cov.values @ w)
        if vol <= 0:
            return 1e6
        return -(port_ret - risk_free_rate) / vol

    cons = [{'type':'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(min_weight, max_weight)] * n
    x0 = np.array([1/n]*n)
    try:
        res = minimize(neg_sharpe, x0, bounds=bounds, constraints=cons)
        w = np.clip(res.x, 0, 1)
        if w.sum()==0:
            return pd.Series(np.ones(n)/n, index=mu.index)
        return pd.Series(w / w.sum(), index=mu.index)
    except Exception:
        return pd.Series(np.ones(n)/n, index=mu.index)

# ----------------------------
# UI: upload / inputs
# ----------------------------
st.sidebar.header("Inputs")
uploaded = st.sidebar.file_uploader("Upload trades.csv (required)", type=["csv"])
monthly_sip_default = 15000
monthly_sip = st.sidebar.number_input("Monthly SIP (₹)", min_value=0, value=monthly_sip_default, step=500)

show_backtest = st.sidebar.checkbox("Show Backtest & Equity Curve", value=True)
show_drawdown = st.sidebar.checkbox("Show Max Drawdown", value=True)
show_heatmap = st.sidebar.checkbox("Show Monthly Return Heatmap", value=True)
show_returns = st.sidebar.checkbox("Show ETFs 6-month Returns", value=True)
show_ef = st.sidebar.checkbox("Show Efficient Frontier & Tangent Allocation", value=True)
show_monte = st.sidebar.checkbox("Show Monte Carlo Simulations", value=True)
show_rebalance = st.sidebar.checkbox("Show Manual Rebalance & Adhoc Allocation", value=True)

# --------------
# Load trades
# --------------
trades = safe_read_trades(uploaded)
if trades.empty:
    st.error("Please upload a valid trades.csv (columns: date,etf,price,units,amount,type) — or place history/trades.csv locally.")
    st.stop()
if not ensure_trade_cols(trades):
    st.error("trades.csv missing required columns. Required: date,etf,price,units,amount,type")
    st.stop()

# Normalize ETF tickers from trades
trade_etfs = sorted(trades["etf"].astype(str).unique().tolist())

st.info(f"Detected ETFs: {', '.join(trade_etfs)} (will fetch last 6 months from yfinance)")

# --------------
# Fetch prices (6 months)
# --------------
with st.spinner("Downloading price data from Yahoo Finance (6 months)..."):
    prices = download_prices_for(trade_etfs, period="6mo")

if prices.empty:
    st.error("No price data downloaded. Aborting.")
    st.stop()

# forward/backfill small holes
prices = prices.sort_index().ffill().bfill()

# --------------
# Holdings & current values
# --------------
trades["date"] = pd.to_datetime(trades["date"], dayfirst=True, errors="coerce")
units = trades.groupby("etf")["units"].sum()
amounts = trades.groupby("etf")["amount"].sum()
avg_cost = (amounts / units).replace([np.inf, -np.inf], 0).fillna(0)

latest = prices.index.max()
# Align current prices
curr_px = prices.loc[latest, prices.columns.intersection(units.index)].reindex(units.index).fillna(method="ffill").fillna(0)

vals = (units.reindex(curr_px.index).fillna(0) * curr_px)
current_val = vals.sum()
total_inv = amounts.sum()
first_trade = trades["date"].min()
n_months = ((latest.year - first_trade.year)*12 + latest.month - first_trade.month - (latest.day < first_trade.day))
if n_months <= 0: n_months = 1
fd_actual = monthly_sip * (((1 + 0.065/12)**n_months - 1)/(0.065/12))  # approximate FD for SIP
gain_actual = current_val - total_inv

# XIRR
cashflows = []
for _, row in trades.iterrows():
    d = row["date"]
    amt = -float(row["amount"])
    cashflows.append((d, amt))
cashflows.append((latest, float(current_val)))
xirr_val = compute_xirr(cashflows)
xirr_display = f"{xirr_val*100:.2f}%" if np.isfinite(xirr_val) else "—"

# -----------------------
# Portfolio Overview UI
# -----------------------
st.subheader("🗃 Portfolio Overview")

df = pd.DataFrame({
    "Units": units.reindex(curr_px.index).fillna(0).astype(int),
    "Avg Cost (₹)": avg_cost.reindex(curr_px.index).fillna(0).round(2),
    "Curr Price (₹)": curr_px.round(2),
    "Value (₹)": vals.round(0)
})
df["Invested (₹)"] = (df["Units"] * df["Avg Cost (₹)"]).round(0)
df["Gain/Loss (₹)"] = (df["Value (₹)"] - df["Invested (₹)"]).round(0)
df["Gain/Loss %"] = ((df["Gain/Loss (₹)"] / df["Invested (₹)"]).replace([np.inf,-np.inf],0).fillna(0)*100).round(2)
df["% of Portfolio"] = (df["Value (₹)"] / df["Value (₹)"].sum() * 100).round(2)

def color_gain(v):
    return f"color: {'green' if v>0 else 'red' if v<0 else 'black'}"

col1, col2 = st.columns((3,2))
with col1:
    st.dataframe(
        df.style.format({
            "Avg Cost (₹)":"₹{:,.2f}",
            "Curr Price (₹)":"₹{:,.2f}",
            "Value (₹)":"₹{:,.0f}",
            "Invested (₹)":"₹{:,.0f}",
            "Gain/Loss (₹)":"₹{:,.0f}",
            "Gain/Loss %":"{:.2f}%",
            "% of Portfolio":"{:.2f}%"
        })
        .applymap(lambda v: "font-weight: bold" if v==df["Value (₹)"].max() else "")
        .applymap(lambda v: "text-align: right")
        .applymap(lambda v: "", subset=None)
        .apply(lambda s: ["color: green" if x>0 else "color: red" if x<0 else "" for x in s], subset=["Gain/Loss (₹)"])
        .set_properties(**{"text-align":"center"})
    , use_container_width=True)

with col2:
    slice_vals = df["Value (₹)"][df["Value (₹)"]>0]
    if slice_vals.sum() > 0:
        fig_pie = go.Figure(go.Pie(labels=slice_vals.index, values=slice_vals.values, hole=0.4, textinfo="label+percent"))
        fig_pie.update_layout(title="Portfolio Allocation", margin=dict(t=20,b=20))
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.write("No positive holdings to show allocation.")

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("💸 Total Invested", f"₹{total_inv:,.0f}")
m2.metric("📈 Current Value", f"₹{current_val:,.0f}")
m3.metric("🏦 FD Benchmark (simple)", f"₹{fd_actual:,.0f}")
m4.metric("📊 Net Gain", f"₹{gain_actual:,.0f}")
m5.metric("🚀 XIRR", xirr_display)

# -----------------------
# Backtest (6 months) + equity curve
# -----------------------
from math import isnan
def simulate_sip_backtest_6m(prices_df, sip_amount, months=6):
    prices_df = prices_df.dropna(how="all", axis=1)
    if prices_df.empty:
        return pd.DataFrame()
    # find month starts
    start = prices_df.index.max() - pd.DateOffset(months=months)
    ms = pd.date_range(start=start.normalize(), end=prices_df.index.max(), freq="MS")
    month_dates = []
    for d in ms:
        pos = prices_df.index.searchsorted(d)
        if pos < len(prices_df.index):
            month_dates.append(prices_df.index[pos])
    units = {c:0.0 for c in prices_df.columns}
    invested = 0.0
    hist = []
    for i, dt in enumerate(month_dates):
        px = prices_df.loc[dt]
        elig = [c for c in prices_df.columns if not pd.isna(px[c])]
        if elig:
            per = sip_amount / len(elig)
            for e in elig:
                units[e] += per / px[e]
        invested += sip_amount
        port_val = sum(units[e]*px[e] for e in prices_df.columns if e in units)
        m = i+1
        fd_val = sip_amount * (((1 + 0.065/12)**m - 1)/(0.065/12))
        hist.append({"Date":dt, "Invested":invested, "Portfolio":port_val, "FD":fd_val})
    dfh = pd.DataFrame(hist).set_index("Date")
    if not dfh.empty:
        dfh["Gain"] = dfh["Portfolio"] - dfh["Invested"]
        yrs = (dfh.index[-1] - dfh.index[0]).days / 365.25
        dfh["CAGR"] = ((dfh["Portfolio"]/dfh["Invested"]) ** (1/yrs) - 1).fillna(0)
    return dfh

if show_backtest:
    st.subheader("📈 Backtest Equity Curve (6 months SIP)")
    bt_df = simulate_sip_backtest_6m(prices, monthly_sip, months=6)
    if bt_df.empty:
        st.write("Not enough data to run backtest.")
    else:
        # plot
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(bt_df.index, bt_df["Invested"], label="Invested", linestyle="--")
        ax.plot(bt_df.index, bt_df["Portfolio"], label="Portfolio")
        ax.plot(bt_df.index, bt_df["FD"], label="FD Benchmark")
        ax.set_title("Backtest Equity Curve")
        ax.legend()
        st.pyplot(fig)

# -----------------------
# Max Drawdown
# -----------------------
def plot_drawdown_from_prices(prices_df):
    norm = prices_df.div(prices_df.iloc[0])
    port = norm.mean(axis=1)  # equal weight portfolio
    cummax = port.cummax()
    drawdown = (port - cummax)/cummax
    return drawdown

if show_drawdown:
    st.subheader("📉 Max Drawdown Chart")
    dd = plot_drawdown_from_prices(prices)
    fig, ax = plt.subplots(figsize=(10,3))
    ax.fill_between(dd.index, dd*100, 0, color='red', alpha=0.3)
    ax.plot(dd.index, dd*100, color='red')
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Portfolio Drawdown (equal-weighted)")
    st.pyplot(fig)

# -----------------------
# Monthly Return Heatmap
# -----------------------
if show_heatmap:
    st.subheader("📅 Monthly Return Heatmap")
    monthly = prices.resample("M").last().pct_change().dropna()
    avg = monthly.mean(axis=1)
    heat = (monthly*100).T
    # pivot by year/month for axis readability
    heat_df = monthly.copy()
    heat_df["Year"] = heat_df.index.year
    heat_df["Month"] = heat_df.index.month
    pivot = heat_df.groupby(["Year","Month"]).mean()["Avg"] if "Avg" in heat_df.columns else None

    # We'll show a simple heatmap of monthly returns for the equal-weight portfolio
    port_monthly = monthly.mean(axis=1) * 100
    if not port_monthly.empty:
        fig, ax = plt.subplots(figsize=(10,3))
        ax.bar(port_monthly.index.strftime("%Y-%m"), port_monthly.values, color=['green' if v>=0 else 'red' for v in port_monthly.values])
        ax.set_xticklabels(port_monthly.index.strftime("%Y-%m"), rotation=45)
        ax.set_ylabel("Monthly Return (%)")
        ax.set_title("Portfolio Monthly Returns (equal-weight)")
        st.pyplot(fig)
    else:
        st.write("Not enough monthly data for heatmap.")

# -----------------------
# All ETFs by 6-month Return (bars)
# -----------------------
if show_returns:
    st.subheader("📊 ETFs by 6-month Return")
    # compute returns over available window (start->end)
    if prices.shape[0] < 2:
        st.write("Not enough data.")
    else:
        total_ret = ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100
        total_ret = total_ret.sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, max(3, len(total_ret)*0.45)))
        colors = ['green' if v>=0 else 'red' for v in total_ret.values]
        bars = ax.barh(total_ret.index, total_ret.values, color=colors)
        for bar, val in zip(bars, total_ret.values):
            xpos = val + (0.5 if val>=0 else -0.5)
            ha = 'left' if val>=0 else 'right'
            ax.text(xpos, bar.get_y()+bar.get_height()/2, f"{val:.2f}%", va='center', ha=ha)
        ax.xaxis.set_major_formatter(PercentFormatter())
        ax.invert_yaxis()
        ax.set_title("6-month Returns (best → worst)")
        st.pyplot(fig)

# -----------------------
# Efficient Frontier + CML + Tangent allocation
# -----------------------
if show_ef:
    st.subheader("📐 Efficient Frontier + CML (annualized)")
    # Use returns data for available ETFs
    rets = prices.pct_change().dropna()
    if rets.empty or rets.shape[1] < 2:
        st.write("Not enough ETF return series for efficient frontier.")
    else:
        ann_mu = rets.mean() * 252
        ann_cov = rets.cov() * 252
        n_assets = len(ann_mu)
        # dynamic bounds
        min_w = max(0.01, 0.5 / n_assets)
        max_w = min(0.9, 2.0 / n_assets)

        # Monte Carlo portfolios
        sims = 2000
        results = np.zeros((3, sims))
        for i in range(sims):
            w = np.random.dirichlet(np.ones(n_assets))
            port_ret = w @ ann_mu
            port_vol = np.sqrt(w @ ann_cov.values @ w)
            port_sr = (port_ret - 0.065) / port_vol if port_vol>0 else 0
            results[:,i] = [port_vol, port_ret, port_sr]
        ef = pd.DataFrame(results.T, columns=["Volatility","Return","Sharpe"])
        best = ef.loc[ef["Sharpe"].idxmax()]

        # Tangent portfolio via constrained optimizer (min/max weights)
        tangent_w = optimize_weights(rets, risk_free_rate=0.065, min_weight=min_w, max_weight=max_w)
        if len(tangent_w)==0:
            tangent_w = pd.Series(np.ones(n_assets)/n_assets, index=ann_mu.index)
        best_ret = tangent_w @ ann_mu
        best_vol = np.sqrt(tangent_w @ ann_cov.values @ tangent_w)
        best_sr = (best_ret - 0.065) / best_vol if best_vol>0 else 0.0

        # Plot EF + CML
        vol_lin = np.linspace(0, ef.Volatility.max()*1.1, 100)
        ret_lin = 0.065 + best_sr * vol_lin

        fig, ax = plt.subplots(figsize=(9,5))
        sc = ax.scatter(ef.Volatility, ef.Return, c=ef.Sharpe, cmap="viridis", alpha=0.5)
        ax.plot(vol_lin, ret_lin, color="darkorange", linewidth=2, label="CML")
        ax.axhline(0.065, color='gray', linestyle='--', label=f"FD 6.5%")
        ax.scatter([best_vol], [best_ret], marker="*", c="red", s=200, label=f"Tangent (Sharpe {best_sr:.2f})")
        ax.set_xlabel("Annual Volatility")
        ax.set_ylabel("Annual Return")
        ax.set_title("Efficient Frontier with CML")
        ax.legend()
        fig.colorbar(sc, label="Sharpe")
        st.pyplot(fig)

        # Tangent allocation bar chart
        st.subheader("🎯 Tangent Portfolio Allocation (Max Sharpe)")
        alloc = (tangent_w * 100).sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(8,3))
        alloc.plot.bar(ax=ax2)
        ax2.set_ylabel("Weight (%)")
        ax2.set_ylim(0, max(alloc.max()*1.2, 0.2))
        ax2.set_title("Asset Weights of Max-Sharpe Portfolio")
        st.pyplot(fig2)

# -----------------------
# Manual Rebalance & Ad-hoc allocation
# -----------------------
if show_rebalance:
    st.subheader("🔄 Manual Rebalance (target = tangent weights)")

    if st.button("Compute Rebalance (use current holdings from trades.csv)"):
        latest_dt = latest
        curr_px2 = curr_px.copy()
        # current units from trades' cumulative sum by ETF
        curr_units = trades.groupby("etf")["units"].sum().reindex(curr_px2.index).fillna(0).astype(int)

        # recompute tangent weights on available assets subset
        rets_sub = prices[curr_px2.index].pct_change().dropna()
        n = rets_sub.shape[1]
        min_w = max(0.01, 0.5 / max(1,n))
        max_w = min(0.9, 2.0 / max(1,n))
        tangent_w2 = optimize_weights(rets_sub, risk_free_rate=0.065, min_weight=min_w, max_weight=max_w)
        if tangent_w2.empty:
            tangent_w2 = pd.Series(np.ones(n)/n, index=rets_sub.columns)

        curr_port_val = (curr_units * curr_px2).sum()
        desired_units = (tangent_w2 * curr_port_val / curr_px2).round(0).astype(int)
        delta_units = desired_units - curr_units
        costs = (delta_units * curr_px2).round(0)

        df_rb = pd.DataFrame({
            "Current Units": curr_units,
            "Target Units": desired_units,
            "Δ Units": delta_units,
            "Price (₹)": curr_px2.round(2),
            "Cost (₹)": costs
        }).loc[tangent_w2.index]
        st.table(df_rb.style.format({"Price (₹)":"₹{:,.2f}", "Cost (₹)":"₹{:,.0f}"}))

    st.subheader("➕ Ad-Hoc Top-Up Allocation (use tangent weights)")
    extra_cash = st.number_input("Extra Cash Amount (₹)", min_value=0, step=1000, value=monthly_sip)
    if st.button("Compute Allocation"):
        # reuse tangent_w2 if exists else recompute
        rets_sub = prices.pct_change().dropna()
        tangent_w3 = optimize_weights(rets_sub, risk_free_rate=0.065,
                                      min_weight=max(0.01, 0.5/rets_sub.shape[1]),
                                      max_weight=min(0.9, 2.0/rets_sub.shape[1]))
        alloc_units = (tangent_w3 * extra_cash / curr_px).fillna(0).astype(int)
        df_extra = pd.DataFrame({
            "Weight (%)": (tangent_w3*100).round(2),
            "Alloc Cash (₹)": (alloc_units * curr_px).round(0).astype(int),
            "Alloc Units": alloc_units
        }).loc[tangent_w3.index]
        st.dataframe(df_extra.style.format({"Weight (%)":"{:.2f}%","Alloc Cash (₹)":"₹{:,.0f}"}))

# -----------------------
# Future SIP projections
# -----------------------
st.subheader("🔮 Future SIP Projections (based on backtest CAGR)")
perf_bt = bt_df if 'bt_df' in locals() else pd.DataFrame()
if not perf_bt.empty:
    last_cagr = perf_bt["CAGR"].iloc[-1] if "CAGR" in perf_bt.columns else 0.0
else:
    last_cagr = 0.05  # fallback
horizons = [5,10,15]
proj_cols = ["Total Invested","Portfolio","FD","Gain","CAGR","Beat_FD"]
proj_df = {}
for yrs in horizons:
    months = yrs*12
    inv = monthly_sip * months
    base_r = float(last_cagr) if not pd.isna(last_cagr) else 0.0
    if base_r > -1:
        monthly_rate = (1+base_r)**(1/12) - 1
        val = monthly_sip * (((1+monthly_rate)**months - 1) / monthly_rate) if monthly_rate!=0 else monthly_sip*months
    else:
        val = 0
    fd_mr = (1+0.065)**(yrs)
    proj_df[yrs] = {"Total Invested":inv, "Portfolio":val, "FD":fd_mr, "Gain":val-inv, "CAGR":compute_cagr(inv,val,months)*100 if val>0 else 0.0, "Beat_FD": val>fd_mr}

for yrs in horizons:
    st.markdown(f"**Over next {yrs} years**")
    d = proj_df[yrs]
    table = pd.DataFrame([d]).T
    st.table(table.style.format({"Total Invested":"₹{:,.0f}","Portfolio":"₹{:,.0f}","FD":"₹{:,.0f}","Gain":"₹{:,.0f}","CAGR":"{:.2f}%"}))

# -----------------------
# Monte Carlo Simulation (3 horizons with user SIP inside expander)
# -----------------------
if show_monte:
    st.subheader("🔮 Monte Carlo Simulation")
    mc_sip = st.number_input("Monte Carlo — Monthly SIP (₹)", min_value=0, value=int(monthly_sip), step=500)
    mc_sims = st.number_input("Number of simulations", min_value=100, max_value=20000, value=2000, step=100)
    years_list = [5,10,15]
    for years in years_list:
        with st.expander(f"📈 {years}-Year Monte Carlo (SIP ₹{mc_sip}/mo)", expanded=False):
            # get portfolio-level daily return stats
            daily = prices.pct_change().dropna()
            if daily.empty:
                st.write("Insufficient return series for Monte Carlo.")
                continue
            port_ret = daily.mean(axis=1)
            mu = port_ret.mean()
            sigma = port_ret.std()
            months = years*12
            results = np.zeros((mc_sims, months+1))
            results[:,0] = current_val
            for i in range(mc_sims):
                for t in range(1, months+1):
                    shock = np.random.normal(mu*21, sigma*np.sqrt(21))
                    results[i,t] = results[i,t-1]*(1+shock) + mc_sip
            final = results[:,-1]
            median = np.median(final)
            fd_proj = current_val * ((1+0.065)**years) + mc_sip * (((1+0.065)**years - 1)/(0.065/12))
            beat_fd = np.mean(final > fd_proj) * 100
            fig, ax = plt.subplots(figsize=(9,4))
            p5, p25, p50, p75, p95 = np.percentile(results, [5,25,50,75,95], axis=0)
            months_arr = np.arange(months+1)/12
            ax.plot(months_arr, p50, label="Median")
            ax.fill_between(months_arr, p25, p75, alpha=0.2, label="25–75%")
            ax.fill_between(months_arr, p5, p95, alpha=0.1, label="5–95%")
            ax.axhline(fd_proj, linestyle="--", color="gray", label=f"FD {years}y (₹{fd_proj:,.0f})")
            ax.set_title(f"Monte Carlo – {years} Years")
            ax.set_xlabel("Years")
            ax.set_ylabel("Portfolio value (₹)")
            ax.legend()
            st.pyplot(fig)
            st.write(f"- Median final: ₹{median:,.0f}")
            st.write(f"- FD benchmark (approx): ₹{fd_proj:,.0f}")
            st.write(f"- % sims beating FD: {beat_fd:.1f}%")

# -----------------------
# Required SIP for target goal
# -----------------------
st.subheader("🎯 Required SIP for Target Goal")
target = st.number_input("Desired final portfolio value (₹)", min_value=0, step=100000, value=10_000_000)
cagr_options = {"-10%": max(last_cagr - 0.10, -0.99), "Actual": last_cagr, "+10%": last_cagr + 0.10}
horiz = [5,10,15]
req = {}
for label,cagr in cagr_options.items():
    req[label] = {}
    for yrs in horiz:
        months = yrs*12
        if cagr > -1:
            mrate = (1+cagr)**(1/12)-1
            sip = target * mrate / ((1+mrate)**months - 1) if mrate!=0 else target/months
            req[label][yrs] = sip
        else:
            req[label][yrs] = np.nan

req_df = pd.DataFrame(req)
req_df.index.name = "Years"
st.table(req_df.applymap(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—"))

st.caption("Public edition: uses Yahoo Finance for price data (6 months). For private edition you can replace price source with cached parquet or NSE fetcher.")
