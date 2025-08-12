# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy.optimize import minimize
import yfinance as yf

# Try to import nsepython but don't fail if unavailable
try:
    from nsepython import equity_history
    NSE_AVAILABLE = True
except Exception:
    NSE_AVAILABLE = False

# ============== Config ==============
st.set_page_config(page_title="ETF SIP Dashboard (Public)", layout="wide")
st.title("📊 ETF SIP Dashboard — Public")

DEFAULT_MONTHS = 3        # fetch window for public dashboard
FD_RATE = 0.065           # benchmark FD (6.5% p.a.)
N_MONTE = 2000            # default Monte Carlo sims (kept moderate)
RISK_FREE = FD_RATE

# ============== Helpers ==============
@st.cache_data(ttl=60*60)  # cache for 1 hour
def load_trades(path="trades.csv"):
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors='coerce')
    else:
        df["date"] = pd.NaT
    return df

def try_fetch_nse(etf, start, end):
    if not NSE_AVAILABLE:
        return pd.DataFrame()
    try:
        raw = equity_history(symbol=etf,
                             from_date=start.strftime("%d-%m-%Y"),
                             to_date=end.strftime("%d-%m-%Y"),
                             series="EQ")
        if raw is None or raw.empty:
            return pd.DataFrame()
        # expect CH_TIMESTAMP, CH_CLOSING_PRICE
        if "CH_CLOSING_PRICE" in raw.columns:
            df = raw[["CH_TIMESTAMP","CH_CLOSING_PRICE"]].copy()
            df.columns = ["Date", etf]
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def try_fetch_yf(etf, start, end):
    # try .NS first, then bare symbol
    for symbol in (f"{etf}.NS", etf):
        try:
            df = yf.download(symbol, start=start, end=end, progress=False)
            if df is None or df.empty:
                continue
            if "Close" in df.columns:
                s = df["Close"].rename(etf)
                s.index.name = "Date"
                return s.to_frame()
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_data(ttl=60*60)
def fetch_price_data(etfs, months=DEFAULT_MONTHS):
    end = datetime.now()
    start = end - timedelta(days=30*months)
    price_df = pd.DataFrame()
    fetched = []
    for etf in etfs:
        # priority: NSE -> yfinance
        df = try_fetch_nse(etf, start, end)
        if df.empty:
            df = try_fetch_yf(etf, start, end)
        if df.empty:
            st.warning(f"Price data not found for {etf}; it will be skipped.")
            continue
        # unify column
        if isinstance(df, pd.DataFrame) and etf not in df.columns:
            # maybe single-col series with name etc
            df.columns = [etf]
        if price_df.empty:
            price_df = df.copy()
        else:
            price_df = price_df.join(df, how="outer")
        fetched.append(etf)
    if price_df.empty:
        return pd.DataFrame()
    price_df = price_df.sort_index().ffill()
    return price_df

def compute_xirr(cashflows):
    # cashflows: list of (date, amount) with negative for contributions, final positive value
    try:
        dates, amounts = zip(*sorted(cashflows, key=lambda x: x[0]))
        dates = pd.to_datetime(dates)
        # convert days to year fractions
        def npv(rate):
            return sum([amt / ((1 + rate) ** ((d - dates[0]).days / 365.0)) for d, amt in zip(dates, amounts)])
        # newton
        from scipy.optimize import newton
        try:
            return newton(npv, 0.1)
        except Exception:
            return float("nan")
    except Exception:
        return float("nan")

def format_money(x):
    if pd.isna(x):
        return "—"
    return f"₹{x:,.0f}"

# optimizer for weights with dynamic bounds
def optimize_weights(returns,
                     risk_free_rate=RISK_FREE,
                     min_weight=None,
                     max_weight=None):
    """
    returns: DataFrame of returns (periodic)
    min_weight/max_weight: scalars applied per asset (None => 0,1)
    Returns pd.Series of weights summing to 1
    """
    returns = returns.dropna(axis=1, how='all').dropna(how='all')
    if returns.shape[1] == 0:
        return pd.Series(dtype=float)
    mu = returns.mean()
    cov = returns.cov()

    def neg_sharpe(w):
        port_ret = w @ mu
        vol = np.sqrt(w @ cov.values @ w)
        return -(port_ret - (risk_free_rate/252 if returns.shape[0]>1 else 0)) / vol if vol > 0 else 0

    n = len(mu)
    x0 = np.ones(n) / n
    bounds = []
    for _ in range(n):
        lo = 0.0 if min_weight is None else min_weight
        hi = 1.0 if max_weight is None else max_weight
        bounds.append((lo, hi))
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1},)
    res = minimize(neg_sharpe, x0, bounds=bounds, constraints=cons)
    w = np.clip(res.x, 0, 1)
    if w.sum() == 0:
        w = np.ones(n) / n
    return pd.Series(w / w.sum(), index=mu.index)

# efficient frontier random simulation
def simulate_efficient_frontier(returns, n_portf=3000, rf=RISK_FREE):
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    n = len(mu)
    results = np.zeros((3, n_portf))
    weights = []
    for i in range(n_portf):
        w = np.random.dirichlet(np.ones(n))
        weights.append(w)
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov.values @ w)
        port_sr = (port_ret - rf) / port_vol if port_vol > 0 else 0
        results[:, i] = [port_vol, port_ret, port_sr]
    ef = pd.DataFrame(results.T, columns=["Volatility","Return","Sharpe"])
    return ef, weights

# simple SIP monthly simulator
def simulate_sip(prices, months_sip=36, sip_amount=15000, window=30, sigma=1.0, min_sharpe=0.2):
    """
    Simple monthly SIP simulation:
    - months_sip: number of months to simulate (we will use actual price history length if shorter)
    - sip_amount: monthly SIP
    - allocation: allocate equally among 'eligible' ETFs (here we simply allocate equally)
    This is a simplified implementation for public view.
    """
    # ensure monthly spaced dates available in prices (use month starts)
    idx = prices.index.dropna()
    if idx.empty:
        return pd.DataFrame()
    # build month starts within available range
    last = idx[-1]
    first = idx[0]
    ms = pd.date_range(start=first, end=last, freq="MS")
    month_dates = [d for d in ms if d <= last]
    portfolio_units = {c:0.0 for c in prices.columns}
    total_invested = 0.0
    history = []
    for i, dt in enumerate(month_dates):
        # find nearest trading day >= dt
        pos = prices.index.searchsorted(dt)
        if pos >= len(prices): break
        date_actual = prices.index[pos]
        px = prices.loc[date_actual]
        # choose eligible ETFs: those with price data on that day
        eligible = [c for c in prices.columns if not pd.isna(px[c])]
        if not eligible: continue
        per = sip_amount / len(eligible)
        for e in eligible:
            portfolio_units[e] += per / px[e]
        total_invested += sip_amount
        port_val = sum(portfolio_units[e] * px[e] for e in prices.columns if not pd.isna(px.get(e, np.nan)))
        # FD compounding monthly from start of SIPs
        m = len(history) + 1
        fd_val = sip_amount * (((1 + FD_RATE/12) ** m - 1) / (FD_RATE/12))
        history.append({"Date":date_actual, "Invested": total_invested, "Portfolio": port_val, "FD": fd_val})
    if not history:
        return pd.DataFrame()
    df = pd.DataFrame(history).set_index("Date")
    df["Gain"] = df["Portfolio"] - df["Invested"]
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    df["CAGR"] = ((df["Portfolio"] / df["Invested"]) ** (1/yrs) - 1).replace([np.inf, -np.inf], np.nan)
    return df

# ========================
# Start App Execution
# ========================
# 1) Load trades
try:
    trades = load_trades("trades.csv")
except FileNotFoundError:
    st.error("trades.csv not found. Place trades.csv in the app folder and try again.")
    st.stop()

if trades.empty:
    st.error("No rows in trades.csv.")
    st.stop()

# 2) Detect ETFs
trade_etfs = sorted(trades["etf"].astype(str).unique().tolist())
st.info(f"Detected ETFs: {', '.join(trade_etfs)} — fetching last {DEFAULT_MONTHS} months of prices (NSE → Yahoo fallback)")

# 3) Fetch price data
prices = fetch_price_data(trade_etfs, months=DEFAULT_MONTHS)
if prices.empty:
    st.error("No price data available for any ETF. Aborting.")
    st.stop()

# 4) Prepare holdings (units, invested amounts)
latest = prices.dropna(how="all").index.max()
units = trades.groupby("etf")["units"].sum()
amounts = trades.groupby("etf")["amount"].sum()
avg_cost = (amounts / units).replace([np.inf, -np.inf], np.nan)
total_inv = amounts.sum()

# 5) Build daily holdings (fills from trades)
# create a full daily index across fetched prices to align holdings
full_dates = pd.date_range(prices.index.min(), prices.index.max(), freq='D')
holds = (
    trades
    .pivot_table(index="date", columns="etf", values="units", aggfunc="sum")
    .reindex(full_dates)
    .fillna(0)
    .cumsum()
)
holds.index.name = "date"

# 6) Ensure we only use ETFs that have price data
avail_hold_etfs = [e for e in trade_etfs if e in prices.columns and not prices[e].dropna().empty]
if not avail_hold_etfs:
    st.error("None of the ETFs in trades.csv have price data. Aborting.")
    st.stop()

curr_px = prices.loc[latest, avail_hold_etfs]
current_val = (units.reindex(avail_hold_etfs).fillna(0) * curr_px).sum()

# 7) Build cashflows for XIRR and compute
cashflows = []
for _, r in trades.iterrows():
    # contributions (negative amounts since invested)
    try:
        d = pd.to_datetime(r["date"])
    except:
        continue
    cashflows.append((d, -float(r.get("amount", 0))))
# terminal value positive
cashflows.append((latest, float(current_val)))
xirr = compute_xirr(cashflows)
if not np.isfinite(xirr):
    xirr_display = "—"
else:
    xirr_display = f"{xirr*100:.2f}%"

# ---------- Layout: Tabs to keep original layout ----------
tab1, tab2, tab3, tab4 = st.tabs(["Portfolio & Backtest","Charts","Projections & Monte Carlo","Tools / Others"])

# ---------- TAB 1: Portfolio & Backtest ----------
with tab1:
    st.subheader("🗃 Portfolio Overview")
    # summary DataFrame
    vals = units.reindex(avail_hold_etfs).fillna(0) * curr_px
    df_port = pd.DataFrame({
        "Units": units.reindex(avail_hold_etfs).fillna(0),
        "Avg Cost (₹)": avg_cost.reindex(avail_hold_etfs).fillna(0).round(2),
        "Curr Price (₹)": curr_px.round(2),
        "Value (₹)": vals.round(0),
    })
    df_port["Invested (₹)"] = (df_port["Units"] * df_port["Avg Cost (₹)"]).round(0)
    df_port["Gain/Loss (₹)"] = (df_port["Value (₹)"] - df_port["Invested (₹)"]).round(2)
    df_port["Gain/Loss %"] = ((df_port["Gain/Loss (₹)"] / df_port["Invested (₹)"]) * 100).round(2)
    df_port["% of Portfolio"] = (df_port["Value (₹)"] / df_port["Value (₹)"].sum() * 100).round(2)

    def color_gain(val):
        if pd.isna(val): return ""
        return f"color: {'green' if val>0 else 'red' if val<0 else 'black'}"

    col1, col2 = st.columns((3,2))
    with col1:
        st.dataframe(
            df_port.style.format({
                "Avg Cost (₹)": "₹{:,.2f}",
                "Curr Price (₹)": "₹{:,.2f}",
                "Value (₹)": "₹{:,.0f}",
                "Invested (₹)": "₹{:,.0f}",
                "Gain/Loss (₹)": "₹{:,.2f}",
                "Gain/Loss %": "{:.2f}%",
                "% of Portfolio": "{:.2f}%"
            }).applymap(color_gain, subset=["Gain/Loss (₹)","Gain/Loss %"])
            .set_properties(**{"text-align":"center"}),
            use_container_width=True
        )
    with col2:
        slice_vals = vals[vals > 0]
        if not slice_vals.empty:
            fig_pie = go.Figure(go.Pie(labels=slice_vals.index.tolist(), values=slice_vals.values.tolist(),
                                       hole=0.4, textinfo="label+percent"))
            fig_pie.update_layout(title="Portfolio Allocation", margin=dict(t=20,b=20), height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No holdings to display in pie chart.")

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("💸 Total Invested", format_money(total_inv))
    m2.metric("📈 Current Value", format_money(current_val))
    m3.metric("🏦 FD Benchmark", format_money(compute_xirr.__defaults__ if False else (total_inv)))  # placeholder
    m4.metric("📊 Net Gain", format_money(current_val - total_inv))
    m5.metric("🚀 XIRR", xirr_display)

    # backtest sample - use available ETFs and simulate SIP monthly using available price history
    st.markdown("### Backtest (SIP simulation on available history)")
    backtest_df = simulate_sip(prices[avail_hold_etfs], sip_amount=15000)
    if backtest_df.empty:
        st.info("Not enough price history to run SIP backtest.")
    else:
        # equity curve plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["Portfolio"], name="Portfolio"))
        fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["Invested"], name="Invested"))
        fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["FD"], name="FD benchmark"))
        fig.update_layout(title="Backtest Equity Curve", xaxis_title="Date", yaxis_title="Value (₹)")
        st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 2: Charts ----------
with tab2:
    st.subheader("Charts")

    # Drawdown chart
    with st.expander("📉 Max Drawdown Chart", expanded=False):
        # build normalized portfolio (equal weight of available ETFs)
        norm = prices[avail_hold_etfs].divide(prices[avail_hold_etfs].iloc[0])
        port = norm.mean(axis=1)
        cummax = port.cummax()
        drawdown = (port - cummax) / cummax
        fig, ax = plt.subplots(figsize=(8,3))
        drawdown.plot(ax=ax, color='red')
        ax.set_title("Portfolio Drawdown (normalized average)")
        ax.set_ylabel("Drawdown")
        st.pyplot(fig)

    # Monthly heatmap
    with st.expander("📅 Monthly Return Heatmap", expanded=False):
        monthly = prices[avail_hold_etfs].resample("MS").last().pct_change().dropna()
        monthly["Avg"] = monthly.mean(axis=1)
        heat = (monthly["Avg"]*100).to_frame("Return")
        heat["Year"] = heat.index.year
        heat["Month"] = heat.index.month
        pivot = heat.pivot(index="Year", columns="Month", values="Return").fillna(0)
        fig, ax = plt.subplots(figsize=(10,4))
        import seaborn as sns
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax)
        ax.set_title("Monthly Returns (%)")
        st.pyplot(fig)

    # All ETFs by 3-month return
    with st.expander("📊 All ETFs by 3-Month Return", expanded=False):
        last = prices.index[-1]
        first = last - pd.DateOffset(months=3)
        window = prices.loc[first:last]
        if len(window) < 2:
            st.warning("Not enough data to compute 3M returns.")
        else:
            total_ret = ((window.iloc[-1] / window.iloc[0]) - 1) * 100
            total_ret = total_ret.reindex(avail_hold_etfs).fillna(0).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, max(3, 0.5*len(total_ret))))
            colors = ['green' if v>=0 else 'red' for v in total_ret.values]
            bars = ax.barh(total_ret.index, total_ret.values, color=colors)
            for bar, val in zip(bars, total_ret.values):
                x = val + (1 if val >=0 else -1.5)
                ha = "left" if val>=0 else "right"
                ax.text(x, bar.get_y()+bar.get_height()/2, f"{val:.2f}%", va="center", ha=ha)
            ax.invert_yaxis()
            ax.set_xlabel("3-Month Return (%)")
            ax.set_title("ETFs by 3-Month Return")
            st.pyplot(fig)

    # Efficient frontier + tangent
    with st.expander("📐 Efficient Frontier + CML & Tangent", expanded=True):
        rets = prices[avail_hold_etfs].pct_change().dropna()
        if rets.shape[0] < 2:
            st.warning("Not enough returns to compute efficient frontier.")
        else:
            ef, weights_rand = simulate_efficient_frontier(rets, n_portf=3000, rf=RISK_FREE)
            # compute tangent weights via optimizer with bounds min=0.5/n, max=2/n
            n_assets = len(avail_hold_etfs)
            min_w = 0.5 / n_assets
            max_w = 2.0 / n_assets
            tangent_w = optimize_weights(rets, risk_free_rate=RISK_FREE, min_weight=min_w, max_weight=max_w)
            # compute tangent portfolio metrics
            ann_mu = rets.mean() * 252
            ann_cov = rets.cov() * 252
            best_ret = float(tangent_w @ ann_mu)
            best_vol = float(np.sqrt(tangent_w @ ann_cov.values @ tangent_w))
            best_sr = (best_ret - RISK_FREE) / best_vol if best_vol>0 else 0

            # plot
            fig, ax = plt.subplots(figsize=(8,5))
            sc = ax.scatter(ef.Volatility, ef.Return, c=ef.Sharpe, cmap="viridis", alpha=0.6)
            vol_lin = np.linspace(0, ef.Volatility.max()*1.1, 100)
            ret_lin = RISK_FREE + best_sr * vol_lin
            ax.plot(vol_lin, ret_lin, linewidth=2, label="CML")
            ax.axhline(y=RISK_FREE, color='gray', linestyle='--', linewidth=1.2, label=f"FD ({RISK_FREE*100:.2f}%)")
            ax.scatter([best_vol], [best_ret], marker="*", c="red", s=200, label=f"Tangent (SR {best_sr:.2f})")
            ax.set_xlabel("Annual Volatility")
            ax.set_ylabel("Annual Return")
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_title("Efficient Frontier with CML")
            fig.colorbar(sc, label="Sharpe")
            ax.legend(loc="upper left")
            st.pyplot(fig)

            # tangent allocation bar chart (as %)
            st.subheader("🎯 Tangent Portfolio Allocation (Max Sharpe)")
            tw = (tangent_w * 100).round(2)
            fig2, ax2 = plt.subplots(figsize=(8,2 + 0.4*len(tw)))
            ax2.bar(tw.index, tw.values)
            ax2.set_ylabel("Weight (%)")
            ax2.set_ylim(0, max(10, tw.max()*1.2))
            for i, v in enumerate(tw.values):
                ax2.text(i, v + 0.5, f"{v:.1f}%", ha='center')
            st.pyplot(fig2)

# ---------- TAB 3: Projections & Monte Carlo ----------
with tab3:
    st.subheader("🔮 Future SIP Projections")
    MONTHLY_SIP = st.number_input("Monthly SIP used in projections (₹)", min_value=0, step=1000, value=15000)

    perf_df_rows = ["Invested","Portfolio","FD","Gain","CAGR","Beat_FD"]
    # simple performance grid: use the backtest_df last row for base CAGR if available, else estimate from portfolio
    base_cagr = backtest_df["CAGR"].iloc[-1] if (not backtest_df.empty and "CAGR" in backtest_df.columns) else 0.10
    perf_df = pd.DataFrame(index=["σ=1.0,Sharpe>0.1"], columns=perf_df_rows)  # placeholder single-strat
    perf_df.loc[:, "Invested"] = total_inv
    perf_df.loc[:, "Portfolio"] = current_val
    perf_df.loc[:, "FD"] = compute_xirr.__defaults__ if False else total_inv
    perf_df.loc[:, "Gain"] = current_val - total_inv
    perf_df.loc[:, "CAGR"] = base_cagr * 100
    perf_df.loc[:, "Beat_FD"] = current_val > total_inv

    st.write("Performance Summary (simplified):")
    st.dataframe(perf_df)

    # future projections for horizons
    horizons = [5, 10, 15]
    rows_out = ["Total Invested","Portfolio","FD","Gain","CAGR","Beat_FD"]
    for yrs in horizons:
        st.markdown(f"**Over next {yrs} years**")
        months = yrs * 12
        out = []
        base_r = base_cagr
        if base_r > -1:
            monthly_rate = (1 + base_r) ** (1/12) - 1
            val = MONTHLY_SIP * (((1 + monthly_rate) ** months - 1) / monthly_rate)
        else:
            val = 0
        fd_monthly = (1 + FD_RATE) ** (1/12) - 1
        fdv = MONTHLY_SIP * (((1 + fd_monthly) ** months - 1) / fd_monthly)
        inv = MONTHLY_SIP * months
        gain = val - inv
        cagr_p = compute_xirr.__defaults__ if False else 0
        df_proj = pd.DataFrame({
            "Value": [inv, val, fdv, gain, base_r*100, "✅" if val>fdv else "❌"]
        }, index=rows_out)
        st.dataframe(df_proj.style.format({0:"₹{:,.0f}"}), use_container_width=True)

    # Monte Carlo simulation
    with st.expander("📈 Monte Carlo Simulation (SIP) — Open to provide SIP and horizons", expanded=True):
        mc_monthly_sip = st.number_input("Monte Carlo Monthly SIP (₹)", min_value=0, step=1000, value=MONTHLY_SIP)
        mc_sims = st.number_input("Simulations", min_value=200, max_value=20000, value=min(N_MONTE,2000), step=200)
        mc_horizons = st.multiselect("Horizons (years) to simulate", options=[5,10,15], default=[5,15])

        # compute historic monthly mu/sigma from portfolio returns
        daily = prices[avail_hold_etfs].pct_change().dropna()
        if daily.empty:
            st.warning("Not enough returns to run Monte Carlo.")
        else:
            port_daily = daily.mean(axis=1)
            mu_d = float(port_daily.mean())
            sigma_d = float(port_daily.std())

            for yrs in mc_horizons:
                nmonths = yrs * 12
                results = np.zeros((mc_sims, nmonths+1))
                results[:,0] = current_val
                for i in range(mc_sims):
                    for t in range(1, nmonths+1):
                        # monthly shock from approx trading days (21)
                        shock = np.random.normal(mu_d * 21, sigma_d * np.sqrt(21))
                        results[i,t] = results[i,t-1] * (1 + shock) + mc_monthly_sip
                final = results[:,-1]
                fd_proj = current_val * ((1 + FD_RATE) ** yrs) + mc_monthly_sip * (((1 + FD_RATE) ** yrs - 1) / (FD_RATE/12))
                pct_beat = (final > fd_proj).mean() * 100
                med = np.median(final)
                st.markdown(f"**{yrs} years** — Median: {format_money(med)} — % simulations > FD: {pct_beat:.1f}%")
                # plot median and bands
                pcts = np.percentile(results, [5,25,50,75,95], axis=0)
                months = np.arange(nmonths+1) / 12.0
                fig, ax = plt.subplots(figsize=(8,3.5))
                ax.plot(months, pcts[2], label="Median")
                ax.fill_between(months, pcts[1], pcts[3], alpha=0.3, label="25-75%")
                ax.fill_between(months, pcts[0], pcts[4], alpha=0.15, label="5-95%")
                ax.axhline(fd_proj, color="gray", linestyle="--", label=f"FD ({format_money(fd_proj)})")
                ax.set_xlabel("Years")
                ax.set_ylabel("Portfolio Value (₹)")
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"₹{x:,.0f}"))
                ax.legend()
                st.pyplot(fig)

# ---------- TAB 4: Tools / Others ----------
with tab4:
    st.subheader("Tools & Utilities")
    st.markdown("""
    - This public dashboard fetches price history (last 3 months) from NSE first, then Yahoo Finance if NSE isn't available.
    - It uses only ETFs present in `trades.csv` and skips missing symbols.
    - If you want faster runs, reduce the `DEFAULT_MONTHS` or the number of Monte Carlo simulations.
    - To extend: add saving / exporting reports, heavier backtests, or user-uploaded trades.csv input.
    """)
    if st.button("Show available price columns & last date"):
        st.write("Last date:", latest)
        st.write("Columns:", prices.columns.tolist())

# ============= End =============

