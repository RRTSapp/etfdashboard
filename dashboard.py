import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from nsepython import equity_history

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="ETF Dashboard", layout="wide")
st.title("📊 ETF Dashboard – Public Version")

# -----------------------------
# Load Trades
# -----------------------------
@st.cache_data
def load_trades():
    df = pd.read_csv("trades.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    return df

trades = load_trades()

# -----------------------------
# Detect ETFs from trades
# -----------------------------
trade_etfs = sorted(trades["etf"].astype(str).unique().tolist())
st.info(f"Detected ETFs: {', '.join(trade_etfs)} (fetching last 3 months)")

# -----------------------------
# Fetch Price Data
# -----------------------------
@st.cache_data
def fetch_price_data(etfs, months=3):
    end = datetime.now()
    start = end - timedelta(days=30 * months)
    price_data = pd.DataFrame()

    for etf in etfs:
        df_prices = pd.DataFrame()

        # Try NSE first
        try:
            nse_df = equity_history(
                symbol=etf,
                from_date=start.strftime("%d-%m-%Y"),
                to_date=end.strftime("%d-%m-%Y"),
                series="EQ"
            )
            if not nse_df.empty and "CH_CLOSING_PRICE" in nse_df.columns:
                df_prices = nse_df[["CH_TIMESTAMP", "CH_CLOSING_PRICE"]].copy()
                df_prices.columns = ["Date", etf]
                df_prices["Date"] = pd.to_datetime(df_prices["Date"])
                df_prices.set_index("Date", inplace=True)
                st.success(f"✅ NSE data loaded for {etf}")
            else:
                raise ValueError("NSE returned empty data")
        except Exception as e:
            st.warning(f"NSE fetch failed for {etf}: {e}")
            # Try Yahoo
            try:
                ticker = f"{etf}.NS"
                ydf = yf.download(ticker, start=start, end=end, progress=False)["Close"]
                if not ydf.empty:
                    df_prices = pd.DataFrame(ydf).rename(columns={"Close": etf})
                    st.info(f"📡 Yahoo Finance data loaded for {etf}")
            except Exception as e2:
                st.error(f"❌ Price fetch failed for {etf}: {e2}")

        if not df_prices.empty:
            if price_data.empty:
                price_data = df_prices
            else:
                price_data = price_data.join(df_prices, how="outer")

    return price_data.sort_index()

prices = fetch_price_data(trade_etfs, months=3)

if prices.empty:
    st.error("❌ No price data available for any ETF. Aborting.")
    st.stop()

# -----------------------------
# Portfolio Summary Calculation
# -----------------------------
latest = prices.dropna(how="all").index.max()
avail_hold_etfs = [e for e in trade_etfs if e in prices.columns and not prices[e].dropna().empty]
curr_px = prices.loc[latest, avail_hold_etfs]

units = trades.groupby("etf")["units"].sum()
avg_cost = trades.groupby("etf").apply(lambda x: np.average(x["price"], weights=x["units"]))
vals = units.reindex(avail_hold_etfs).fillna(0) * curr_px

df = pd.DataFrame({
    "Units": units.reindex(avail_hold_etfs).fillna(0),
    "Avg Cost (₹)": avg_cost.reindex(avail_hold_etfs).fillna(0).round(2),
    "Curr Price (₹)": curr_px.round(2),
    "Value (₹)": vals.round(0),
})

df["Invested (₹)"] = df["Units"] * df["Avg Cost (₹)"]
df["Gain/Loss (₹)"] = (df["Value (₹)"] - df["Invested (₹)"]).round(2)
df["Gain/Loss %"] = ((df["Gain/Loss (₹)"] / df["Invested (₹)"]) * 100).round(2)
df["% of Portfolio"] = (df["Value (₹)"] / df["Value (₹)"].sum() * 100).round(2)

# -----------------------------
# Display Summary Table & Pie
# -----------------------------
col1, col2 = st.columns((3, 2))

with col1:
    def color_gain(val):
        if pd.isna(val):
            return ""
        color = "green" if val > 0 else "red" if val < 0 else "black"
        return f"color: {color}"

    st.dataframe(
        df.style.format({
            "Avg Cost (₹)": "₹{:,.2f}",
            "Curr Price (₹)": "₹{:,.2f}",
            "Value (₹)": "₹{:,.0f}",
            "Invested (₹)": "₹{:,.0f}",
            "Gain/Loss (₹)": "₹{:,.2f}",
            "Gain/Loss %": "{:.2f}%",
            "% of Portfolio": "{:.2f}%"
        }).applymap(color_gain, subset=["Gain/Loss (₹)", "Gain/Loss %"]),
        use_container_width=True,
    )

with col2:
    slice_vals = vals[vals > 0]
    if not slice_vals.empty:
        fig_pie = go.Figure(go.Pie(
            labels=slice_vals.index.tolist(),
            values=slice_vals.values.tolist(),
            hole=0.4,
            textinfo="label+percent",
        ))
        fig_pie.update_layout(title="Portfolio Allocation", margin=dict(t=20, b=20), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("No positive value holdings to plot in pie chart.")

# -----------------------------
# Placeholder for Charts & Backtest
# -----------------------------
