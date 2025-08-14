# dashboard.py
import io
import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import numpy_financial as npf
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")

# ---------------------------------------------
# App config
# ---------------------------------------------
st.set_page_config(page_title="ETF SIP Dashboard (Public)", layout="wide")
st.title("📊 ETF SIP Dashboard (Public)")

# Defaults
DEFAULT_FD_RATE = 0.065   # 6.5% annual
DEFAULT_MONTHLY_SIP = 15000
RISK_FREE = 0.065
LOOKBACK_DAYS = 90        # ~3 months

# ---------------------------------------------
# Helpers
# ---------------------------------------------
def format_money(x, zero_dash=True):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—" if zero_dash else "₹0"
    try:
        return f"₹{x:,.0f}"
    except Exception:
        return "—"

def safe_xirr(cashflows):
    """
    cashflows: list of (date, amount) tuples. Negative for investments, positive for redemption.
    Returns a float (annualized) or None if not solvable.
    """
    if not cashflows:
        return None
    # Convert to days since first date
    cashflows = sorted(cashflows, key=lambda x: x[0])
    t0 = cashflows[0][0]
    amounts = [cf[1] for cf in cashflows]
    days = [(cf[0] - t0).days for cf in cashflows]
    # If all same sign, IRR undefined
    if all(a >= 0 for a in amounts) or all(a <= 0 for a in amounts):
        return None
    # Use IRR on irregular periods → try Newton by converting to periodic with day fractions
    def xnpv(rate):
        return sum(a / ((1 + rate) ** (d/365.0)) for a, d in zip(amounts, days))
    # bracket search then Newton
    try:
        # coarse bracket
        low, high = -0.999, 5.0
        for _ in range(60):
            mid = (low + high) / 2
            val = xnpv(mid)
            if abs(val) < 1e-8:
                return mid
            # move bracket
            val_low = xnpv(low)
            # bisection
            if (val_low > 0 and val < 0) or (val_low < 0 and val > 0):
                high = mid
            else:
                low = mid
        return mid
    except Exception:
        return None

def pct_to_str(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "—"

def compute_fd_value(lumpsum: float, months: int, annual_rate: float) -> float:
    """Future value of a lumpsum compounded monthly at given annual_rate."""
    mrate = annual_rate / 12
    return lumpsum * ((1 + mrate) ** months)

def compute_fd_with_sip(current_value, monthly_sip, years, annual_rate):
    """FD projection with principal (current value + monthly SIP annuity)."""
    mrate = annual_rate / 12
    n = years * 12
    fv_lump = current_value * ((1 + mrate) ** n)
    fv_sip  = monthly_sip * (((1 + mrate) ** n - 1) / mrate)
    return fv_lump + fv_sip

def monthly_heatmap(returns_daily: pd.Series):
    monthly = returns_daily.resample("M").apply(lambda x: (1 + x).prod() - 1)
    df = monthly.to_frame("Return").copy()
    df["Year"] = df.index.year
    df["Month"] = df.index.strftime("%b")
    pivot = df.pivot_table(index="Year", columns="Month", values="Return", aggfunc="mean")
    # reorder months
    months_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = pivot.reindex(columns=[m for m in months_order if m in pivot.columns])
    return pivot

def annualize_daily(mu_d, sigma_d):
    mu_ann = mu_d * 252
    sig_ann = sigma_d * np.sqrt(252)
    return mu_ann, sig_ann

def max_sharpe_weights(mu_vec, cov_mat, rf=RISK_FREE):
    """Max Sharpe using analytic solution (no bounds). Fall back to numeric if needed."""
    try:
        inv = np.linalg.pinv(cov_mat)
        ones = np.ones(len(mu_vec))
        # tangency weights proportional to inv*(mu - rf)
        w = inv @ (mu_vec - rf*ones)
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            w = np.ones_like(w)
        w = w / w.sum()
        return w
    except Exception:
        # fallback equal-weight
        n = len(mu_vec)
        return np.ones(n)/n

def drawdown_curve(series: pd.Series) -> pd.Series:
    cummax = series.cummax()
    dd = series/cummax - 1.0
    return dd

def calc_equity_curve(prices: pd.DataFrame, weights: pd.Series, start_val=100.0) -> pd.Series:
    w = weights.reindex(prices.columns).fillna(0).values
    rets = prices.pct_change().dropna()
    port = (rets @ w) + 1
    equity = pd.Series(np.r_[start_val, start_val*port.cumprod().values], index=prices.index)
    return equity

# ---------------------------------------------
# Upload trades.csv
# ---------------------------------------------
st.sidebar.header("📥 Upload your trades.csv")
trades_file = st.sidebar.file_uploader("Upload trades.csv (columns: date, etf, units, price)", type=["csv"])

if trades_file is None:
    st.info("Upload a **trades.csv** to continue. Expected columns: `date, etf, units, price`.")
    st.stop()

# Read trades
try:
    trades = pd.read_csv(trades_file)
except Exception as e:
    st.error(f"Could not read trades.csv: {e}")
    st.stop()

# Normalize
required_cols = {"date","etf","units","price"}
missing = required_cols - set(c.lower() for c in trades.columns)
if missing:
    st.error(f"trades.csv is missing columns: {', '.join(missing)}")
    st.stop()

trades.columns = [c.lower() for c in trades.columns]
trades["date"] = pd.to_datetime(trades["date"], errors="coerce")
trades = trades.dropna(subset=["date","etf","units","price"])
trades["etf"] = trades["etf"].astype(str).str.strip().str.upper()
trade_etfs = sorted(trades["etf"].unique().tolist())

st.success(f"Detected ETFs: {', '.join(trade_etfs)}")
st.caption("Price data will be fetched for the last ~3 months (90 calendar days). Missing tickers are skipped.")

# ---------------------------------------------
# Fetch last 3 months prices (daily) - yfinance
# ---------------------------------------------
# Replace your previous fetch_prices_3m(...) with this NSE-aware implementation.
# Requires: pip install nsepython

# Paste this into your dashboard.py and call fetch_prices_3m_combined(trade_etfs, lookback_days=90)

import os
import time
import traceback
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

def fetch_prices_3m_combined(etfs, lookback_days=90, verbose=True):
    """
    Robust fetcher:
      1) Try nsepython.equity_history(symbol, 'EQ', start_date, end_date)
      2) parse many possible return shapes (DataFrame, list, dict)
      3) fallback to yfinance attempts: SYMBOL.NS then SYMBOL
      4) log / save raw responses for debugging if parsing fails
    Returns a DataFrame indexed by datetime with one column per ETF (forward-filled).
    """
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=lookback_days + 10)  # a bit of buffer

    # helper to normalize various raw outputs into a pd.Series of closes
    def _normalize_raw_to_series(raw, symbol):
        """Return pd.Series indexed by datetime, name = symbol, or None."""
        try:
            if raw is None:
                return None

            # If it's already a DataFrame
            if isinstance(raw, pd.DataFrame):
                df = raw.copy()
            else:
                # Maybe it's list-of-dicts or dict-of-lists or dict
                try:
                    df = pd.DataFrame(raw)
                except Exception:
                    return None

            if df.empty:
                return None

            # 1) find date-like column
            date_col = None
            for c in df.columns:
                cn = str(c).lower()
                if any(k in cn for k in ("date","timestamp","time","ch_timestamp","trade_date")):
                    date_col = c
                    break
            if date_col is None:
                # try to parse first column as date
                for c in df.columns:
                    try:
                        _ = pd.to_datetime(df[c].iloc[0])
                        date_col = c
                        break
                    except Exception:
                        continue
            if date_col is None:
                return None

            # 2) find close-like column
            close_col = None
            closenames = [c for c in df.columns if any(k in str(c).lower() for k in ("close","closing","last","ch_closing","close_price"))]
            if closenames:
                close_col = closenames[0]
            else:
                # use the last numeric column not equal to date_col
                numeric_cols = [c for c in df.columns if c != date_col and np.issubdtype(df[c].dtype, np.number)]
                if numeric_cols:
                    close_col = numeric_cols[-1]

            if close_col is None:
                return None

            # build series
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            df = df.sort_values(by=date_col)
            s = pd.Series(pd.to_numeric(df[close_col], errors="coerce").values,
                          index=pd.to_datetime(df[date_col]),
                          name=symbol).dropna()
            if s.empty:
                return None
            # resample daily and forward-fill for alignment
            s = s.resample("D").ffill()
            return s
        except Exception:
            return None

    # Try NSE first (if available)
    try:
        from nsepython import equity_history
        nse_available = True
    except Exception as e:
        nse_available = False
        if verbose:
            st.warning("nsepython not installed or import failed; skipping NSE fetch. Install via `pip install nsepython` to enable NSE fetching.")
    
    frames = []
    succeeded = []
    failed = []
    for symbol in etfs:
        ser = None
        # 1) NSE path
        if nse_available:
            try:
                # call equity_history(symbol, series, start_date, end_date)
                # use 'EQ' series for equity/etf
                raw = equity_history(symbol, "EQ", start_dt.strftime("%d-%m-%Y"), end_dt.strftime("%d-%m-%Y"))
                # normalize
                ser = _normalize_raw_to_series(raw, symbol)
                if ser is None:
                    # save raw repr for debugging
                    rawfile = f"debug_raw_{symbol}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.txt"
                    try:
                        with open(rawfile, "w", encoding="utf-8") as fh:
                            fh.write("REPR OF RAW FROM NSE:\n\n")
                            fh.write(repr(raw))
                        if verbose:
                            st.warning(f"NSE data fetch returned no usable rows for {symbol} — raw saved to {rawfile}")
                    except Exception:
                        if verbose:
                            st.warning(f"NSE returned unusable rows for {symbol}; could not save raw output.")
                else:
                    frames.append(ser.rename(symbol))
                    succeeded.append((symbol, "NSE"))
                    if verbose:
                        st.success(f"NSE data OK for {symbol}.")
            except Exception as exc:
                failed.append((symbol, f"NSE_exc: {str(exc)[:200]}"))
                if verbose:
                    st.warning(f"NSE data fetch failed for {symbol}: {exc}")
                # continue to fallback
        # 2) fallback: yfinance attempts
        if ser is None:
            # try two symbol forms
            tried = []
            for sym in (f"{symbol}.NS", symbol):
                try:
                    df = yf.download(sym, start=start_dt, end=end_dt + timedelta(days=1), interval="1d", progress=False, auto_adjust=True)
                    if isinstance(df, pd.DataFrame) and "Close" in df.columns and not df["Close"].dropna().empty:
                        s = df["Close"].copy().rename(symbol).dropna()
                        s = s.resample("D").ffill()
                        frames.append(s)
                        succeeded.append((symbol, f"YF:{sym}"))
                        if verbose:
                            st.success(f"yfinance OK for {symbol} using {sym}")
                        ser = s
                        break
                except Exception as exc:
                    tried.append((sym, str(exc)[:200]))
                    continue
            if ser is None and verbose:
                st.warning(f"Price data not found for {symbol} via NSE or yfinance (tried NSE + { [f'{symbol}.NS', symbol] }).")

    if not frames:
        if verbose:
            st.error("No price data downloaded for any ETF. Aborting.")
        return pd.DataFrame()

    df_all = pd.concat(frames, axis=1).sort_index().ffill()
    # trim to lookback_days exactly
    cutoff = pd.Timestamp(date.today() - timedelta(days=lookback_days))
    df_all = df_all[df_all.index >= cutoff]
    # remove columns with all NaNs
    df_all = df_all.loc[:, df_all.notna().any(axis=0)]
    return df_all

    
prices = fetch_prices_3m_nse(trade_etfs)
if prices.empty:
    st.error("No price data downloaded for any ETF. Aborting.")
    st.stop()

avail_etfs = [c for c in prices.columns if c in trade_etfs]
if not avail_etfs:
    st.error("Trades detected, but none have price data. Aborting.")
    st.stop()

prices = prices[avail_etfs].ffill()
latest_date = prices.index.max()
st.caption(f"Prices range: {prices.index.min().date()} → {latest_date.date()}")

# ---------------------------------------------
# Current holdings & metrics from trades
# ---------------------------------------------
grouped = trades.groupby("etf", as_index=True).agg(
    units=("units","sum"),
    invested=("price", lambda s: (s*trades.loc[s.index,"units"]).sum())
)
# Weighted avg cost per ETF
avg_cost = []
for etf in grouped.index:
    sub = trades[trades["etf"]==etf]
    inv = (sub["price"]*sub["units"]).sum()
    u = sub["units"].sum()
    avg_cost.append(inv/u if u!=0 else np.nan)
grouped["avg_cost"] = avg_cost

units = grouped["units"]
avg_cost_s = grouped["avg_cost"]

curr_px = prices.loc[latest_date, avail_etfs]
vals = units.reindex(avail_etfs).fillna(0) * curr_px.reindex(avail_etfs).fillna(0)
current_val = float(vals.sum())
total_inv = float((avg_cost_s.reindex(avail_etfs).fillna(0) * units.reindex(avail_etfs).fillna(0)).sum())

# Build XIRR cashflows from trades + current portfolio value as redemption
cf = []
for _, row in trades.iterrows():
    # cash outflow (buy) = negative
    cf.append((row["date"].to_pydatetime(), -float(row["price"])*float(row["units"])))
# Add final "sale" today = current portfolio value
cf.append((latest_date.to_pydatetime(), current_val))
xirr = safe_xirr(cf)  # None if invalid

# ---------------------------------------------
# Tabs
# ---------------------------------------------
tab1, tab2, tab3 = st.tabs(["📒 Portfolio & Backtest", "📈 Charts", "🔮 Future Projections"])

# ---------------------------------------------
# TAB 1: Portfolio & Backtest
# ---------------------------------------------
with tab1:
    colA, colB, colC = st.columns(3)
    colA.metric("💸 Total Invested", format_money(total_inv))
    colB.metric("📈 Current Value", format_money(current_val))
    colC.metric("🚀 XIRR", pct_to_str(xirr) if xirr is not None else "—")

    # Per-ETF summary table
    df = pd.DataFrame({
        "Units":           units.reindex(avail_etfs).fillna(0),
        "Avg Cost (₹)":    avg_cost_s.reindex(avail_etfs).fillna(0).round(2),
        "Curr Price (₹)":  curr_px.round(2),
        "Value (₹)":       vals.round(0),
    })
    df["Invested (₹)"]   = (df["Units"] * df["Avg Cost (₹)"]).round(0)
    df["Gain/Loss (₹)"]  = (df["Value (₹)"] - df["Invested (₹)"]).round(2)
    df["Gain/Loss %"]    = np.where(df["Invested (₹)"]>0,
                                    (df["Gain/Loss (₹)"]/df["Invested (₹)"])*100, np.nan).round(2)
    df["% of Portfolio"] = np.where(df["Value (₹)"].sum()>0,
                                    (df["Value (₹)"]/df["Value (₹)"].sum()*100).round(2), 0.0)

    c1, c2 = st.columns((3,2))
    with c1:
        styled = (df.style
                    .format({
                        "Avg Cost (₹)":     "₹{:,.2f}",
                        "Curr Price (₹)":   "₹{:,.2f}",
                        "Value (₹)":        "₹{:,.0f}",
                        "Invested (₹)":     "₹{:,.0f}",
                        "Gain/Loss (₹)":    "₹{:,.2f}",
                        "Gain/Loss %":      "{:.2f}%",
                        "% of Portfolio":   "{:.2f}%"
                    })
                    .apply(lambda s: ['color: green' if v>0 else ('color: red' if v<0 else '') 
                                      for v in df["Gain/Loss (₹)"]], axis=0, subset=["Gain/Loss (₹)"])
                    .set_properties(**{"text-align":"center"})
                 )
        st.dataframe(styled, use_container_width=True, height=380)

    with c2:
        slice_vals = df.loc[df["Value (₹)"]>0, "Value (₹)"]
        if not slice_vals.empty:
            fig_pie = go.Figure(go.Pie(
                labels=slice_vals.index.tolist(),
                values=slice_vals.values.tolist(),
                hole=0.4,
                textinfo="label+percent",
            ))
            fig_pie.update_layout(title="Portfolio Allocation", margin=dict(t=20,b=20), height=320)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No positive holdings to display.")

    # Simple backtest (equal-weight over available ETFs)
    with st.expander("▶ Backtest: Equal-Weight over last ~3 months", expanded=False):
        eq_w = pd.Series(1/len(avail_etfs), index=avail_etfs)
        equity = calc_equity_curve(prices, eq_w, start_val=100)
        dd = drawdown_curve(equity)

        c3, c4 = st.columns(2)
        with c3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity (EW)"))
            fig.update_layout(title="Equity Curve", height=300, margin=dict(t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown"))
            fig2.update_layout(title="Max Drawdown (EW)", height=300, margin=dict(t=40,b=20), yaxis_tickformat=".1%")
            st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------
# TAB 2: Charts
# ---------------------------------------------
with tab2:
    st.subheader("Charts")

    # Equity Curve (EW) & Drawdown again for convenience
    eq_w = pd.Series(1/len(avail_etfs), index=avail_etfs)
    equity = calc_equity_curve(prices, eq_w, start_val=100)
    dd = drawdown_curve(equity)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity (EW)"))
        fig.update_layout(title="Equity Curve (EW)", height=300, margin=dict(t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown"))
        fig2.update_layout(title="Drawdown (EW)", height=300, margin=dict(t=40,b=20), yaxis_tickformat=".1%")
        st.plotly_chart(fig2, use_container_width=True)

    # Monthly Return Heatmap (EW)
    port_daily = prices.pct_change().dropna().mean(axis=1)
    heat = monthly_heatmap(port_daily)
    if not heat.empty:
        fig3 = px.imshow(
            heat,
            text_auto=".0%",
            aspect="auto",
            color_continuous_scale="RdYlGn",
            origin="upper"
        )
        fig3.update_layout(title="Monthly Return Heatmap (EW portfolio)", height=360, margin=dict(t=40,b=20))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Not enough data for a monthly heatmap.")

    # All ETFs by 3-month return
    with st.expander("▶ All ETFs by ~3-month return", expanded=True):
        three_mo_returns = {}
        for c in avail_etfs:
            s = prices[c].dropna()
            if len(s) > 5:
                three_mo_returns[c] = s.iloc[-1]/s.iloc[0] - 1
        if three_mo_returns:
            r = pd.Series(three_mo_returns).sort_values(ascending=False)
            fig4 = px.bar(r, x=r.index, y=r.values, text=[f"{v*100:.1f}%" for v in r.values])
            fig4.update_layout(title="ETFs by ~3-Month Return", yaxis_tickformat=".1%", height=360, margin=dict(t=40,b=20))
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No sufficient data to compute ~3-month returns.")

    # Efficient Frontier + CML + Tangent Portfolio
    with st.expander("▶ Efficient Frontier + CML + Tangent Portfolio", expanded=False):
        daily = prices.pct_change().dropna()
        if daily.shape[1] >= 2 and len(daily) > 10:
            mu_d = daily.mean().values
            cov_d = daily.cov().values
            mu_a, _ = annualize_daily(mu_d, daily.std().values)  # only mu_a used here
            cov_a = cov_d * 252

            # Frontier points
            n_pts = 60
            rng = np.linspace(0.0, 1.0, n_pts)
            ef_x, ef_y = [], []
            cols = list(prices.columns)
            # Create random portfolios for frontier cloud
            np.random.seed(42)
            W = []
            for _ in range(1000):
                w = np.random.rand(len(cols))
                w = w / w.sum()
                W.append(w)
            W = np.array(W)
            rets = W @ mu_a
            vols = np.sqrt(np.einsum('ij,jk,ik->i', W, cov_a, W))
            sharpe = (rets - RISK_FREE) / np.where(vols==0, np.nan, vols)

            # Tangent weights
            w_tan = max_sharpe_weights(mu_a, cov_a, rf=RISK_FREE)
            ret_tan = w_tan @ mu_a
            vol_tan = np.sqrt(w_tan @ cov_a @ w_tan)
            sharpe_tan = (ret_tan - RISK_FREE) / vol_tan if vol_tan>0 else np.nan

            # Capital Market Line
            cml_x = np.linspace(0, max(vols.max()*1.1, vol_tan*1.2), 50)
            cml_y = RISK_FREE + sharpe_tan * cml_x

            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=vols, y=rets, mode="markers", name="Random Portfolios", opacity=0.45))
            fig5.add_trace(go.Scatter(x=[vol_tan], y=[ret_tan], mode="markers+text", name="Tangent (Max Sharpe)",
                                      text=["Tangent"], textposition="top center", marker=dict(size=12)))
            fig5.add_trace(go.Scatter(x=cml_x, y=cml_y, mode="lines", name="CML"))
            fig5.update_layout(title="Efficient Frontier + CML", xaxis_title="Volatility (σ, annualized)",
                               yaxis_title="Return (annualized)", height=420, margin=dict(t=50,b=20))
            st.plotly_chart(fig5, use_container_width=True)

            # Show tangent portfolio weights
            w_series = pd.Series(w_tan, index=cols).sort_values(ascending=False)
            st.write("**Tangent Portfolio (Max Sharpe) Weights:**")
            st.dataframe(w_series.to_frame("Weight").style.format("{:.2%}"), use_container_width=True)
        else:
            st.info("Need at least 2 ETFs and ~2 weeks of data to plot the frontier.")

# ---------------------------------------------
# TAB 3: Future Projections
# ---------------------------------------------
with tab3:
    st.subheader("Future Projections")

    MONTHLY_SIP = st.number_input("Enter Monthly SIP Amount (₹)", min_value=0, value=DEFAULT_MONTHLY_SIP, step=1000)
    FD_RATE = st.number_input("FD Annual Rate", min_value=0.0, max_value=0.20, value=DEFAULT_FD_RATE, step=0.005, format="%.3f")

    # Monte Carlo and projections are based on the portfolio daily returns
    port_daily = prices.pct_change().dropna().mean(axis=1)
    if port_daily.empty:
        st.info("Not enough return history for projections.")
    else:
        c1, c2, c3 = st.columns(3)
        for yrs, c in zip([5, 10, 15], [c1, c2, c3]):
            with c:
                st.markdown(f"**{yrs}-Year Monte Carlo**")
                # Estimate mu/sigma from equal-weight daily portfolio
                mu = port_daily.mean()
                sigma = port_daily.std()
                n_months = yrs * 12
                sims = 4000

                results = np.zeros((sims, n_months + 1))
                results[:, 0] = current_val
                for i in range(sims):
                    for t in range(1, n_months + 1):
                        shock = np.random.normal(mu * 21, sigma * np.sqrt(21))  # 21 trading days ≈ month
                        results[i, t] = results[i, t - 1] * (1 + shock) + MONTHLY_SIP

                final_vals = results[:, -1]
                fd_proj = compute_fd_with_sip(current_val, MONTHLY_SIP, yrs, FD_RATE)
                beat_fd = np.mean(final_vals > fd_proj) * 100

                st.write(f"💰 **Monthly SIP**: {format_money(MONTHLY_SIP)}")
                st.write(f"🧮 **Total Invested**: {format_money(MONTHLY_SIP*12*yrs)}")
                st.write(f"📈 **Median value**: {format_money(np.median(final_vals))}")
                st.write(f"🏦 **FD Benchmark ({yrs}Y @ {FD_RATE*100:.1f}%)**: {format_money(fd_proj)}")
                st.write(f"✅ **% beating FD**: {beat_fd:.1f}%")

        st.markdown("---")
        st.subheader("Required SIP for Target Goal")
        target = st.number_input("Enter desired final portfolio value (₹)", min_value=0, step=100000, value=10_00_000)
        # approximate CAGR from last 3 months → scale to annual
        ann_mu = port_daily.mean() * 252
        cagr_scenarios = {
            "–10% CAGR": max(ann_mu - 0.10, -0.99),
            "Actual CAGR": ann_mu,
            "+10% CAGR": ann_mu + 0.10
        }
        horizons = [5, 10, 15]
        req = pd.DataFrame(index=horizons, columns=cagr_scenarios.keys())
        for yrs in horizons:
            months = yrs * 12
            for label, cagr in cagr_scenarios.items():
                if cagr > -1:
                    mrate = (1 + cagr)**(1/12) - 1
                    sip = target * mrate / ((1 + mrate)**months - 1)
                else:
                    sip = np.nan
                req.at[yrs, label] = sip
        req.index.name = "Years"
        st.table(req.applymap(lambda x: format_money(x) if pd.notna(x) else "—"))

