# app.py
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image
import hashlib
import base64

# ---------- Page config and styling ----------
st.set_page_config(page_title="CrowdPitch Pro — Final", page_icon="🚀", layout="wide")

# Custom CSS for polished look
st.markdown(
    """
    <style>
    .css-1d391kg {padding-top:1rem;}  /* adjust top padding */
    .stApp { background: linear-gradient(180deg,#0f1724,#111827); color: #e5e7eb; }
    .block-container { padding: 1.75rem 2rem; }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); 
            border-radius: 12px; padding: 1rem; box-shadow: 0 6px 24px rgba(2,6,23,0.6); }
    .big-btn > button { background-color: #06b6d4; color: #042f33; border-radius: 10px; padding: 0.45rem 0.75rem; }
    .small-muted { color: #9ca3af; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

sns.set_style("darkgrid")
plt.rcParams["figure.dpi"] = 100

# ---------- Utilities ----------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def save_image_to_bytes(img_file) -> bytes:
    # returns bytes suitable for st.image via BytesIO
    img = Image.open(img_file).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def init_state():
    # users: dict username -> {password_hash, role, wallet(if investor)}
    if "users" not in st.session_state:
        st.session_state.users = {}
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "pitches" not in st.session_state:
        st.session_state.pitches = []  # list of pitch dicts
    if "investments" not in st.session_state:
        st.session_state.investments = []  # list of {investor, pitch_name, amount}
init_state()

# ---------- Auth - Signup/Login ----------
def signup(username, password, role):
    if username in st.session_state.users:
        st.warning("Username already exists. Try another.")
        return False
    ph = hash_password(password)
    user = {"password": ph, "role": role}
    if role == "Investor":
        user["wallet"] = 10000.0  # default wallet in INR
    st.session_state.users[username] = user
    st.success("Signup successful! You can now login.")
    return True

def login(username, password):
    user = st.session_state.users.get(username)
    if not user:
        st.error("No such user. Please signup.")
        return False
    if user["password"] != hash_password(password):
        st.error("Incorrect password.")
        return False
    st.session_state.current_user = {"username": username, "role": user["role"]}
    st.success(f"Logged in as {username} ({user['role']})")
    return True

def logout():
    st.session_state.current_user = None
    st.success("Logged out.")

# ---------- Layout: Header ----------
col1, col2 = st.columns([3,1])
with col1:
    st.title("🚀 CrowdPitch Pro — Final Submission")
    st.markdown("**A polished demo:** upload pitch, ML forecasting (ARIMA), investor view, wallet & invest.")
with col2:
    if st.session_state.current_user:
        u = st.session_state.current_user
        st.markdown(f"**{u['username']}**")
        st.markdown(f"_{u['role']}_")
        if st.button("Logout"):
            logout()
    else:
        st.markdown("Not signed in")

st.markdown("---")

# ---------- Sidebar: Auth & Quick Info ----------
with st.sidebar:
    st.markdown("## Account")
    if not st.session_state.current_user:
        auth_tab = st.radio("Action", ["Login", "Signup"], index=0)
        if auth_tab == "Signup":
            su_user = st.text_input("Username (signup)", key="su_user")
            su_pass = st.text_input("Password", type="password", key="su_pass")
            su_role = st.selectbox("Role", ["Startup", "Investor"], key="su_role")
            if st.button("Create account"):
                if su_user and su_pass:
                    signup(su_user, su_pass, su_role)
                else:
                    st.warning("Enter username & password")
        else:
            li_user = st.text_input("Username (login)", key="li_user")
            li_pass = st.text_input("Password", type="password", key="li_pass")
            if st.button("Login"):
                if li_user and li_pass:
                    login(li_user, li_pass)
                else:
                    st.warning("Enter username & password")
        st.markdown("---")
        st.markdown("### Demo accounts")
        st.markdown("- investor_demo / pass123 (investor)")
        st.markdown("- startup_demo / pass123 (startup)")
        # create demo users if not exist
        if "investor_demo" not in st.session_state.users:
            signup("investor_demo", "pass123", "Investor")
        if "startup_demo" not in st.session_state.users:
            signup("startup_demo", "pass123", "Startup")
    else:
        u = st.session_state.current_user
        if u["role"] == "Investor":
            wallet = st.session_state.users[u["username"]].get("wallet", 0.0)
            st.markdown("### Wallet")
            st.markdown(f"**₹ {wallet:,.2f}**")
            st.markdown("---")
            st.markdown("Quick actions")
            if st.button("Add ₹1000"):
                st.session_state.users[u["username"]]["wallet"] += 1000.0
                st.success("Added ₹1000 to wallet")

    st.markdown("---")
    st.markdown("About")
    st.markdown("Built with Streamlit • statsmodels (ARIMA) • seaborn • Session-state demo")

# ---------- MAIN: Role Views ----------
if not st.session_state.current_user:
    st.info("Please signup or login from the sidebar. Use demo accounts for quick testing.")
    st.stop()

user = st.session_state.current_user
role = user["role"]
username = user["username"]

# ---------- STARTUP: Create Pitch ----------
if role == "Startup":
    st.header("🏢 Startup Dashboard — Create a Pitch")
    st.markdown("Fill pitch details, upload a CSV of historical metric (date,value), upload image/logo, add pitch video (YouTube).")

    with st.form("pitch_form", clear_on_submit=False):
        pname = st.text_input("Pitch name")
        pshort = st.text_area("Short description (one line)")
        pdesc = st.text_area("Full description")
        uploaded_csv = st.file_uploader("Upload CSV (date,value)", type=["csv"])
        uploaded_img = st.file_uploader("Upload Image (logo/product) PNG/JPG", type=["png","jpg","jpeg"])
        video_link = st.text_input("YouTube video link (embed) — paste full link")
        submit = st.form_submit_button("Create / Save Pitch")

    if submit:
        if not pname or not uploaded_csv:
            st.error("Please provide at least pitch name and CSV.")
        else:
            # process CSV
            try:
                df = pd.read_csv(uploaded_csv)
                df.columns = ["date", "value"]
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")
            except Exception as e:
                st.error(f"CSV error: {e}")
                st.stop()

            img_bytes = None
            if uploaded_img:
                try:
                    img_bytes = save_image_to_bytes(uploaded_img)
                except Exception as e:
                    st.warning("Image upload problem — continuing without image.")
                    img_bytes = None

            # create pitch record
            pitch = {
                "name": pname,
                "short": pshort,
                "desc": pdesc,
                "owner": username,
                "data": df,
                "image": img_bytes,
                "video": video_link,
                "funded": 0.0,
                "investors": []
            }
            st.session_state.pitches.append(pitch)
            st.success(f"Pitch '{pname}' created!")

    st.markdown("---")
    st.subheader("Your pitches")
    mine = [p for p in st.session_state.pitches if p["owner"] == username]
    if not mine:
        st.info("You haven't created any pitches yet.")
    else:
        for p in mine:
            st.markdown(f"### {p['name']}")
            cols = st.columns([1, 3])
            with cols[0]:
                if p["image"]:
                    st.image(p["image"], use_column_width=True)
                else:
                    st.image("https://via.placeholder.com/220x140.png?text=No+Image", use_column_width=True)
                st.markdown(f"**Funded:** ₹{p['funded']:,.2f}")
            with cols[1]:
                st.markdown(p["short"])
                st.write(p["desc"])
                if st.button(f"View / Predict — {p['name']}", key=f"view_{p['name']}"):
                    st.session_state._view_pitch = p["name"]

# ---------- INVESTOR: Browse & Invest ----------
elif role == "Investor":
    st.header("💼 Investor Dashboard — Discover & Invest")
    st.markdown("Browse startup pitches, view predictions, watch video, and invest (simulated wallet).")
    if not st.session_state.pitches:
        st.info("No pitches available yet. Wait for startups to create pitches.")
        st.stop()

    # Search / Filters
    row1, row2 = st.columns([3,1])
    with row1:
        q = st.text_input("Search startups (name or description)")
    with row2:
        sort_by = st.selectbox("Sort by", ["Newest", "Most Funded", "Alphabetical"])
    pitches = st.session_state.pitches.copy()
    if q:
        pitches = [p for p in pitches if q.lower() in p["name"].lower() or q.lower() in p["short"].lower() or q.lower() in p["desc"].lower()]
    if sort_by == "Most Funded":
        pitches = sorted(pitches, key=lambda x: x['funded'], reverse=True)
    elif sort_by == "Alphabetical":
        pitches = sorted(pitches, key=lambda x: x['name'])
    else:
        pitches = list(reversed(pitches))  # newest first

    # Display pitches as cards
    for p in pitches:
        st.markdown("", unsafe_allow_html=True)
        card_cols = st.columns([1.2, 3])
        with card_cols[0]:
            if p["image"]:
                st.image(p["image"], width=180)
            else:
                st.image("https://via.placeholder.com/220x140.png?text=No+Image", width=180)
            st.markdown(f"**Funded:** ₹{p['funded']:,.2f}")
            st.markdown(f"**Investors:** {len(p['investors'])}")
        with card_cols[1]:
            st.markdown(f"### {p['name']}")
            st.markdown(f"_{p['short']}_")
            st.write(p['desc'][:300] + ("..." if len(p['desc'])>300 else ""))

            # Small inline metrics
            df = p["data"]
            latest = df['value'].iloc[-1]
            mean = df['value'].mean()
            growth = (df['value'].iloc[-1] / df['value'].iloc[0] - 1) * 100
            cols = st.columns(4)
            cols[0].metric("Latest", f"{latest:.2f}")
            cols[1].metric("Avg", f"{mean:.2f}")
            cols[2].metric("Growth (%)", f"{growth:.1f}%")
            cols[3].metric("Funded", f"₹{p['funded']:,.0f}")

            # Buttons: View detail and Invest
            b1, b2 = st.columns([1,1])
            if b1.button("View Details", key=f"view_{p['name']}"):
                st.session_state._view_pitch = p['name']
            if b2.button("Invest", key=f"invest_{p['name']}"):
                st.session_state._invest_in = p['name']

        st.markdown("---")

    # Handle view pitch detail
    if "_view_pitch" in st.session_state:
        target = st.session_state._view_pitch
        selected = next((x for x in st.session_state.pitches if x["name"] == target), None)
        if selected:
            st.markdown("## 🔎 Pitch — " + selected["name"])
            details_cols = st.columns([1,2])
            with details_cols[0]:
                if selected["image"]:
                    st.image(selected["image"], use_column_width=True)
                else:
                    st.image("https://via.placeholder.com/320x180.png?text=No+Image", use_column_width=True)
                st.markdown(f"**Owner:** {selected['owner']}")
                st.markdown(f"**Funded:** ₹{selected['funded']:,.2f}")
                st.markdown(f"**Investors:** {len(selected['investors'])}")
                # Video
                if selected['video']:
                    st.markdown("🎥 Video pitch")
                    st.video(selected['video'])
            with details_cols[1]:
                st.markdown(selected["desc"])
                st.markdown("### Historical data & Forecast")
                df = selected["data"]

                # Plot with seaborn + ARIMA forecast (statsmodels)
                fig, ax = plt.subplots(figsize=(9,4))
                sns.lineplot(x="date", y="value", data=df, marker="o", ax=ax, label="Historical")
                try:
                    model = sm.tsa.ARIMA(df["value"], order=(1,1,1))
                    result = model.fit()
                    n = 6
                    forecast = result.forecast(steps=n)
                    conf = result.get_forecast(steps=n).conf_int()
                    future_dates = pd.date_range(df["date"].iloc[-1], periods=n+1, freq="M")[1:]
                    ax.plot(future_dates, forecast, marker="X", linestyle="--", label="Forecast")
                    ax.fill_between(future_dates, conf.iloc[:,0], conf.iloc[:,1], alpha=0.25)
                except Exception as e:
                    st.warning("Forecast failed: " + str(e))
                ax.set_title("Metric (historical + forecast)")
                st.pyplot(fig)

                # Download forecast CSV if available
                if 'forecast' in locals():
                    fdf = pd.DataFrame({
                        "date": future_dates,
                        "forecast": forecast,
                        "lower": conf.iloc[:,0],
                        "upper": conf.iloc[:,1]
                    })
                    csv_bytes = df_to_csv_bytes(fdf)
                    st.download_button("Download forecast (CSV)", data=csv_bytes, file_name=f"{selected['name']}_forecast.csv", mime="text/csv")

            # close view
            if st.button("Close view"):
                del st.session_state._view_pitch

    # Handle invest action
    if "_invest_in" in st.session_state:
        target = st.session_state._invest_in
        selected = next((x for x in st.session_state.pitches if x["name"] == target), None)
        if selected:
            st.markdown(f"## 💸 Invest in {selected['name']}")
            st.write(selected["short"])
            amount = st.number_input("Amount to invest (INR)", min_value=100.0, value=500.0, step=100.0)
            buyer_wallet = st.session_state.users[username]["wallet"]
            st.markdown(f"**Your wallet:** ₹{buyer_wallet:,.2f}")
            if st.button("Confirm Invest"):
                if amount <= 0:
                    st.error("Enter positive amount")
                elif amount > buyer_wallet:
                    st.error("Insufficient wallet balance")
                else:
                    # process
                    st.session_state.users[username]["wallet"] -= amount
                    selected["funded"] += amount
                    selected["investors"].append({"investor": username, "amount": amount})
                    st.session_state.investments.append({"investor": username, "pitch": selected["name"], "amount": amount})
                    st.success(f"Successfully invested ₹{amount:,.0f} in {selected['name']}")
                    # cleanup
                    del st.session_state._invest_in

# ---------- Footer & quick stats ----------
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total Pitches", f"{len(st.session_state.pitches)}")
with c2:
    total_funded = sum(p["funded"] for p in st.session_state.pitches)
    st.metric("Total Funded", f"₹{total_funded:,.0f}")
with c3:
    st.metric("Total Investments", f"{len(st.session_state.investments)}")

st.markdown("<center>Built with ❤️ — Good luck with your submission!</center>", unsafe_allow_html=True)
