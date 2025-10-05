# ==========================================================
# 🌍 Climate Emoji 2.1 — NASA Space Apps Challenge
# ==========================================================
# Author: Muhammad Osama Nusrat
# Purpose: Translate NASA GISTEMP temperature anomalies
#          into human-readable "Earth Moods" 🌡 + 📊 analytics
# ==========================================================

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
from datetime import datetime

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="Climate Emoji – NASA Space Apps Challenge",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# NASA Header Banner
# -----------------------------
st.markdown("""
<h1 style='text-align:center; color:#33CCFF;'>
🌎 Climate Emoji — NASA Space Apps Challenge 2025
</h1>
""", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)

# -----------------------------
# Load NASA GISTEMP Data
# -----------------------------
NASA_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

@st.cache_data(ttl=3600)
def load_nasa_data():
    df = pd.read_csv(NASA_URL, skiprows=1)
    df = df.replace("*******", pd.NA)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_long = df.melt(id_vars=["Year"], var_name="Month", value_name="Anomaly").dropna()
    month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                 "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
    df_long = df_long[df_long["Month"].isin(month_map.keys())]
    df_long["Date"] = pd.to_datetime(
        df_long["Year"].astype(str) + "-" +
        df_long["Month"].map(month_map).astype(str) + "-01"
    )
    df_long = df_long.sort_values("Date")
    return df_long

df = load_nasa_data()

# -----------------------------
# Helper Functions
# -----------------------------
def get_earth_mood(a):
    if a >= 1.5: return "Hot 🔥"
    elif a >= 0.5: return "Warm 🌱"
    elif a >= 0:   return "Stable 🙂"
    else:          return "Cold ❄️"

def get_mood_color(mood):
    return {"Hot 🔥":"#FF4444","Warm 🌱":"#FF8800",
            "Stable 🙂":"#44AA44","Cold ❄️":"#4488FF"}.get(mood,"#999")

def calc_trend(data, years=10):
    recent = data.tail(years*12)
    if len(recent) < 2: return 0
    x = np.arange(len(recent))
    y = recent["Anomaly"].values
    return np.polyfit(x, y, 1)[0]*12

# -----------------------------
# Prepare Data
# -----------------------------
df["Mood"] = df["Anomaly"].apply(get_earth_mood)
df["Mood_Color"] = df["Mood"].apply(get_mood_color)
df["Rolling10yr"] = df["Anomaly"].rolling(120, min_periods=30).mean()

latest = df.iloc[-1]
trend_10 = calc_trend(df,10)
corr = df["Year"].corr(df["Anomaly"])
pred_2030 = np.polyval(np.polyfit(df["Year"], df["Anomaly"], 1), 2030)
pred_2050 = np.polyval(np.polyfit(df["Year"], df["Anomaly"], 1), 2050)

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("🎛️ Climate Controls")
min_date, max_date = df["Date"].min(), df["Date"].max()
start, end = st.sidebar.date_input("Select Date Range",
                                   value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)
mask = (df["Date"]>=pd.to_datetime(start)) & (df["Date"]<=pd.to_datetime(end))
filtered = df.loc[mask]

mood_filter = st.sidebar.multiselect(
    "Filter by Earth's Mood",
    ["Hot 🔥","Warm 🌱","Stable 🙂","Cold ❄️"],
    default=["Hot 🔥","Warm 🌱","Stable 🙂","Cold ❄️"])
filtered = filtered[filtered["Mood"].isin(mood_filter)]

# -----------------------------
# Narrative: Intro
# -----------------------------
st.markdown("""
### 🌡 Why “Climate Emoji”?
We wanted to make NASA’s global temperature data *human*.  
Instead of abstract numbers, each reading becomes an **Earth Mood** — from ❄️ Cold to 🔥 Hot —  
so anyone can instantly feel what the planet is experiencing.
""")

# -----------------------------
# Metrics Dashboard
# -----------------------------
col1,col2,col3,col4,col5 = st.columns(5)
with col1:
    st.metric("Latest Mood", latest["Mood"], f"{latest['Anomaly']:.3f}°C")
with col2:
    st.metric("10-Year Trend", f"{trend_10:.3f}°C/yr",
              "Warming" if trend_10>0 else "Cooling")
with col3:
    st.metric("Correlation (Year vs Temp)", f"{corr:.2f}")
with col4:
    st.metric("Predicted 2030", f"{pred_2030:.2f}°C")
with col5:
    st.metric("Predicted 2050", f"{pred_2050:.2f}°C")

st.markdown("> **Observation:** Global temperature anomalies have remained above baseline since the 1980s — an unmistakable sign of ongoing warming.")

# -----------------------------
# Interactive Climate Plot
# -----------------------------
st.subheader("🌍 Earth's Climate Journey")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=filtered["Date"], y=filtered["Anomaly"],
    mode="lines", name="Temperature Anomaly",
    line=dict(color="white", width=2)))
fig.add_trace(go.Scatter(
    x=df["Date"], y=df["Rolling10yr"],
    mode="lines", name="10-Year Rolling Mean",
    line=dict(color="#FFAA33", width=3, dash="dash")))
for mood in mood_filter:
    mood_data = filtered[filtered["Mood"]==mood]
    if not mood_data.empty:
        fig.add_trace(go.Scatter(
            x=mood_data["Date"], y=mood_data["Anomaly"],
            mode="markers", name=mood,
            marker=dict(color=get_mood_color(mood), size=8, opacity=0.8)))
fig.add_hline(y=0, line_dash="dash", line_color="#888", annotation_text="Baseline (0°C)")
fig.update_layout(
    title="NASA GISTEMP Global Temperature Anomalies (1880–Present)",
    xaxis_title="Year", yaxis_title="Temperature Anomaly (°C)",
    hovermode="x unified",
    plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
    font=dict(color="white"), height=520)
st.plotly_chart(fig, use_container_width=True)
st.markdown("> **Insight:** The planet’s average temperature has climbed steadily, with sharp rises during El Niño years and recent record highs.")

# -----------------------------
# Compare Two Years
# -----------------------------
st.subheader("📆 Compare Two Years")
c1, c2 = st.columns(2)
year1 = c1.slider("Select Year 1", int(df["Year"].min()), int(df["Year"].max()), 1950)
year2 = c2.slider("Select Year 2", int(df["Year"].min()), int(df["Year"].max()), int(df["Year"].max()))
if year1 in df["Year"].values and year2 in df["Year"].values:
    a1 = df[df["Year"]==year1]["Anomaly"].mean()
    a2 = df[df["Year"]==year2]["Anomaly"].mean()
    delta = a2 - a1
    st.info(f"In **{year1}**, Earth was at **{a1:.2f}°C**, compared to **{a2:.2f}°C** in **{year2}** — a change of **{delta:+.2f}°C**.")

# -----------------------------
# Educational Insights
# -----------------------------
with st.expander("📘 What is a Temperature Anomaly?"):
    st.write("""
A temperature anomaly shows how much warmer or cooler a period is compared to NASA’s **1951–1980 baseline average**.  
This removes seasonal variation and reveals long-term climate trends more clearly.
""")

# -----------------------------
# Enhanced Climate Analytics
# -----------------------------
st.subheader("📊 Climate Analytics")

colA, colB = st.columns(2)
with colA:
    st.markdown("#### 🌡️ Temperature Distribution (NASA GISTEMP Anomalies)")

    fig_hist = px.histogram(
        filtered,
        x="Anomaly",
        nbins=40,
        color_discrete_sequence=["#00BFFF"],
        opacity=0.85
    )

    x_sorted = np.sort(filtered["Anomaly"])
    y_density = np.linspace(0, filtered["Anomaly"].count()/5.5, len(x_sorted))

    fig_hist.add_trace(go.Scatter(
        x=x_sorted,
        y=y_density,
        mode="lines",
        line=dict(color="#FF8C00", width=2, dash="dot"),
        name="Approx. density trend"
    ))

    fig_hist.add_vline(
        x=0,
        line_dash="dash",
        line_color="#FFFFFF",
        annotation_text="Baseline (0°C)",
        annotation_position="top right"
    )

    fig_hist.update_layout(
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white", size=13),
        title=dict(font=dict(size=18, color="#33CCFF")),
        xaxis=dict(
            gridcolor="#333",
            title=dict(text="Temperature Anomaly (°C)", font=dict(size=14, color="#99CCFF"))
        ),
        yaxis=dict(
            gridcolor="#333",
            title=dict(text="Number of Months", font=dict(size=14, color="#99CCFF"))
        ),
        bargap=0.05
    )

    fig_hist.add_annotation(
        text="Notice the shift toward warmer anomalies ➡️",
        x=0.3, y=180,
        showarrow=False,
        font=dict(color="#FFD700", size=14)
    )

    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption("This histogram shows how Earth’s temperature distribution has shifted rightward since 1880 — clear evidence of global warming.")

with colB:
    st.markdown("#### 🪞 Earth's Mood Frequency (Selected Period)")
    mood_counts = filtered["Mood"].value_counts()
    fig_pie = px.pie(
        values=mood_counts.values,
        names=mood_counts.index,
        color_discrete_map={
            "Hot 🔥":"#FF4444","Warm 🌱":"#FF8800",
            "Stable 🙂":"#44AA44","Cold ❄️":"#4488FF"
        },
        title="Earth's Mood Distribution"
    )
    fig_pie.update_traces(textinfo="label+percent", textfont_size=14, pull=[0.05]*len(mood_counts))
    fig_pie.update_layout(
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        font=dict(color="white", size=13)
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    st.caption("Each slice shows the percentage of time Earth spent in each mood — from cold and stable to warm and hot.")



# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
**🌍 NASA Space Apps Challenge 2025 — Team Climate Emoji**  
Live data: [NASA GISTEMP](https://data.giss.nasa.gov/gistemp/)  
Made with ❤️ using Streamlit + Plotly + Python
""")
