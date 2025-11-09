# app.py
"""
CrowdPitch Pro — KYC Edition (Enhanced)
- Landing page (login/signup)
- Startup onboarding (more docs + target funding)
- Checker with inline PDF/image preview
- Investor marketplace with search/filters + card grid (Amazon-style)
- Investor dashboard (my investments) and correct wallet handling
- ARIMA-based ROI prediction (6-month average)
"""

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image
import hashlib
import re
from urllib.parse import urlparse
from datetime import datetime
import base64
import html
import textwrap

# ---------- Page config ----------
st.set_page_config(page_title="CrowdPitch Pro — KYC Edition", page_icon="🚀", layout="wide")
sns.set_style("darkgrid")
plt.rcParams["figure.dpi"] = 100

# ---------- CSS ----------
st.markdown(
    """
    <style>
    body { background: linear-gradient(180deg,#071A2A,#071622); color: #e6eef6; }
    .auth-card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); padding:18px; border-radius:10px; box-shadow:0 6px 18px rgba(0,0,0,0.45); }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); border-radius:12px; padding:12px; box-shadow:0 6px 18px rgba(0,0,0,0.45); margin-bottom:14px;}
    .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap:16px; }
    .pitch-card { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.4); transition: 0.3s; }
    .pitch-card:hover { transform: scale(1.02); background: rgba(255,255,255,0.08); }
    .badge { padding:4px 8px; border-radius:8px; font-weight:600; font-size:0.85rem; }
    .badge-approved { background-color:#16a34a; color:white; }
    .badge-pending { background-color:#f59e0b; color:#012; }
    .badge-rejected { background-color:#ef4444; color:white; }
    .progress { height:10px; background: rgba(255,255,255,0.05); border-radius:8px; overflow:hidden; margin-top:8px; }
    .progress > div { background: linear-gradient(90deg,#06b6d4,#0891b2); height:100%; }
    .small-muted { color:#9fb4c9; font-size:0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Utilities ----------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def image_bytes(img_file) -> bytes:
    img = Image.open(img_file).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def safe_read_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df

def predict_forecast_and_roi(df: pd.DataFrame, steps: int = 6):
    try:
        model = sm.tsa.ARIMA(df["value"], order=(1,1,1))
        res = model.fit()
        forecast = res.forecast(steps=steps)
        conf = res.get_forecast(steps=steps).conf_int()
        future_dates = pd.date_range(df["date"].iloc[-1], periods=steps+1, freq="M")[1:]
        latest = df["value"].iloc[-1]
        mean_forecast = np.mean(forecast)
        roi = (mean_forecast / latest - 1) * 100.0
        return future_dates, np.array(forecast), conf, roi
    except Exception:
        return None, None, None, 0.0

def embed_pdf_bytes(pdf_bytes: bytes, width="100%", height="600px"):
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    src = f"data:application/pdf;base64,{b64}"
    html_tag = f'<iframe src="{src}" width="{width}" height="{height}"></iframe>'
    st.components.v1.html(html_tag, height=int(height.replace("px",""))+20)

# ---------- Session init ----------
if "users" not in st.session_state:
    st.session_state.users = {}
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "pitches" not in st.session_state:
    st.session_state.pitches = []
if "investments" not in st.session_state:
    st.session_state.investments = []
if "complaints" not in st.session_state:
    st.session_state.complaints = []
if "page" not in st.session_state:
    st.session_state.page = "home"

# Demo users
if "investor_demo" not in st.session_state.users:
    st.session_state.users["investor_demo"] = {"password": hash_password("pass123"), "role": "Investor", "wallet": 10000.0}
if "startup_demo" not in st.session_state.users:
    st.session_state.users["startup_demo"] = {"password": hash_password("pass123"), "role": "Startup"}
if "checker_agent" not in st.session_state.users:
    st.session_state.users["checker_agent"] = {"password": hash_password("Check@2025!"), "role": "Checker"}

# Keep all other pages (landing, startup, checker, complaints, etc.) the same
# We replace ONLY the investor_page() below 👇

# ---------- Investor Section (Enhanced) ----------
def investor_page(user):
    st.header("💼 Investor Marketplace")

    wallet = st.session_state.users[user["username"]]["wallet"]
    top1, top2, top3 = st.columns([2,3,1])
    with top1:
        st.metric("Wallet", f"₹{wallet:,.2f}")
    with top2:
        search = st.text_input("🔍 Search by company or pitch")
        roi_min, roi_max = st.slider("Predicted ROI (%) range", -50, 200, (-10, 50))
    with top3:
        view_mode = st.selectbox("View", ["Marketplace", "My Investments"], key="inv_view_mode")

    # My Investments View
    if view_mode == "My Investments":
        st.subheader("📊 My Investments")
        invs = [i for i in st.session_state.investments if i["investor"] == user["username"]]
        if not invs:
            st.info("No investments yet.")
        else:
            df = pd.DataFrame(invs)
            df["date"] = pd.to_datetime(df["date"])
            st.metric("Total Invested", f"₹{df['amount'].sum():,.2f}")
            st.metric("Avg Predicted ROI", f"{df['predicted_roi'].mean():.1f}%")
            st.dataframe(df.sort_values("date", ascending=False))
        return

    # Marketplace Grid
    published = [p for p in st.session_state.pitches if p.get("published")]
    if not published:
        st.info("No published pitches yet.")
        return

    for p in published:
        _, _, _, roi = predict_forecast_and_roi(p["data"], steps=6)
        p["_predicted_roi"] = float(roi or 0.0)

    def matches(p):
        if search and search.lower() not in (p["pitch_name"] + p["company_name"]).lower():
            return False
        return roi_min <= p["_predicted_roi"] <= roi_max

    filtered = [p for p in published if matches(p)]
    st.markdown(f"**{len(filtered)} results found**")
    st.markdown('<div class="grid">', unsafe_allow_html=True)

    for p in filtered:
        st.markdown('<div class="pitch-card">', unsafe_allow_html=True)
        if p.get("logo"):
            st.image(p["logo"], width=250)
        else:
            st.image("https://via.placeholder.com/250x150.png?text=No+Logo", width=250)
        st.markdown(f"### {p['pitch_name']}")
        st.caption(p["company_name"])
        st.write(p.get("short", ""))
        st.progress(min(p["funded"]/p["target"], 1.0))
        st.write(f"💰 ₹{p['funded']:,.0f} / ₹{p['target']:,.0f}")
        st.write(f"📈 Predicted ROI: **{p['_predicted_roi']:.1f}%**")

        c1, c2, c3 = st.columns(3)
        if c1.button("Invest", key=f"inv_{p['pitch_name']}"):
            st.session_state["_invest_in"] = p["pitch_name"]
            st.experimental_rerun()
        if c2.button("Analysis", key=f"ana_{p['pitch_name']}"):
            st.session_state["_analysis"] = p["pitch_name"]
        if c3.button("Complain", key=f"compl_{p['pitch_name']}"):
            st.session_state["_complaint_for"] = p["pitch_name"]
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Investment Modal
    if st.session_state.get("_invest_in"):
        pitch_name = st.session_state["_invest_in"]
        sel = next((x for x in st.session_state.pitches if x["pitch_name"] == pitch_name), None)
        if sel:
            st.markdown("---")
            st.subheader(f"Invest in {sel['pitch_name']}")
            amt = st.number_input("Amount (₹)", min_value=100.0, value=500.0)
            if st.button("Confirm Investment"):
                if amt > wallet:
                    st.error("Insufficient wallet balance.")
                else:
                    st.session_state.users[user["username"]]["wallet"] -= amt
                    sel["funded"] += amt
                    st.session_state.investments.append({
                        "investor": user["username"],
                        "pitch": sel["pitch_name"],
                        "amount": amt,
                        "date": datetime.now(),
                        "predicted_roi": sel["_predicted_roi"]
                    })
                    st.success("Investment successful.")
                    st.session_state["_invest_in"] = None
                    st.experimental_rerun()

    # Analysis Modal
    if st.session_state.get("_analysis"):
        sel = next((x for x in st.session_state.pitches if x["pitch_name"] == st.session_state["_analysis"]), None)
        if sel:
            df = sel["data"]
            st.markdown("---")
            st.subheader(f"📊 Analysis — {sel['pitch_name']}")

            # Line chart
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.lineplot(df, x="date", y="value", marker="o", ax=ax)
            ax.set_title("Historical Performance")
            st.pyplot(fig)

            # Side charts
            c1, c2 = st.columns(2)
            with c1:
                df["month"] = df["date"].dt.month
                df["year"] = df["date"].dt.year
                pivot = df.pivot_table(index="year", columns="month", values="value", aggfunc="mean")
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.heatmap(pivot, annot=True, fmt=".1f", ax=ax)
                ax.set_title("Monthly Heatmap")
                st.pyplot(fig)
            with c2:
                df["rolling"] = df["value"].rolling(3).mean()
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(df["date"], df["rolling"], color="orange")
                ax.set_title("3-Month Rolling Average")
                st.pyplot(fig)

            # Correlation + Forecast
            df["lag1"] = df["value"].shift(1)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(df[["value", "lag1"]].corr(), annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Map")
            st.pyplot(fig)

            fdates, fvals, conf, roi = predict_forecast_and_roi(df)
            if fdates is not None:
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(df["date"], df["value"], label="Historical")
                ax.plot(fdates, fvals, linestyle="--", marker="o", label="Forecast")
                if conf is not None:
                    ax.fill_between(fdates, conf.iloc[:,0], conf.iloc[:,1], alpha=0.2)
                ax.legend()
                ax.set_title("6-Month Forecast")
                st.pyplot(fig)
                st.success(f"Predicted ROI: {roi:.2f}%")
            if st.button("Close Analysis"):
                st.session_state["_analysis"] = None

# ---------- Main routing ----------
def main_app():
    user = st.session_state.current_user
    st.sidebar.markdown(f"**Logged in as:** {user['username']} ({user['role']})")
    if st.sidebar.button("Logout"): st.session_state.page, st.session_state.current_user = "home", None; st.rerun()
    if user["role"] == "Investor": investor_page(user)
    elif user["role"] == "Startup": st.write("Startup page remains unchanged.")
    elif user["role"] == "Checker": st.write("Checker page remains unchanged.")

if st.session_state.page == "home" or st.session_state.current_user is None:
    st.write("Landing page (unchanged)")
else:
    main_app()

