# app.py
"""
CrowdPitch Pro — KYC Edition
Landing page (signin / signup) + role-based dashboard
Features: Startup onboarding, manual Checker verification, Investor browsing & investing,
ARIMA forecasting, Complaint box, hidden Checker account, demo users.
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

# ---------- Page config ----------
st.set_page_config(page_title="CrowdPitch Pro — KYC Edition", page_icon="🚀", layout="wide")
sns.set_style("darkgrid")
plt.rcParams["figure.dpi"] = 100

# ---------- Small CSS ----------
st.markdown(
    """
    <style>
    .app-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
    .brand { display:flex; align-items:center; gap:12px; }
    .brand h1 { margin:0; color:#06b6d4; }
    .tag { color:#9fb4c9; margin-top:4px; font-size:0.95rem; }
    .hero { background: linear-gradient(180deg,#071A2A,#071622); padding:28px; border-radius:12px; box-shadow:0 6px 18px rgba(0,0,0,0.5); }
    .auth-card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); padding:18px; border-radius:10px; box-shadow:0 6px 18px rgba(0,0,0,0.45); }
    .small-muted { color:#9fb4c9; font-size:0.92rem; }
    .badge { padding:3px 8px; border-radius:8px; font-weight:600; font-size:0.85rem; }
    .badge-approved { background-color:#16a34a; color:white; }
    .badge-pending { background-color:#f59e0b; color:#012; }
    .badge-rejected { background-color:#ef4444; color:white; }
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

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        domain = domain.lower().replace("www.", "")
        domain = domain.split(':')[0]
        return domain
    except:
        return ""

def email_domain(email: str) -> str:
    parts = email.split("@")
    return parts[1].lower() if len(parts) == 2 else ""

def name_in_filename(name: str, fname: str) -> bool:
    if not name or not fname:
        return False
    n = re.sub(r'[^a-z0-9]', '', name.lower())
    f = re.sub(r'[^a-z0-9]', '', fname.lower())
    return n in f

def status_badge_html(status: str) -> str:
    s = (status or "").lower()
    if s == "approved":
        return '<span class="badge badge-approved">🟢 Approved</span>'
    if s == "rejected":
        return '<span class="badge badge-rejected">🔴 Rejected</span>'
    return '<span class="badge badge-pending">🟡 Pending</span>'

# ---------- Session-state initialization ----------
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
    st.session_state.page = "home"  # 'home' or 'app'

# demo accounts
if "investor_demo" not in st.session_state.users:
    st.session_state.users["investor_demo"] = {"password": hash_password("pass123"), "role": "Investor", "wallet": 10000.0}
if "startup_demo" not in st.session_state.users:
    st.session_state.users["startup_demo"] = {"password": hash_password("pass123"), "role": "Startup"}
# hidden checker
if "checker_agent" not in st.session_state.users:
    st.session_state.users["checker_agent"] = {"password": hash_password("Check@2025!"), "role": "Checker"}

# ---------- Auth functions ----------
def signup(username: str, password: str, role: str):
    if not username or not password:
        st.warning("Provide username and password.")
        return False
    if username in st.session_state.users:
        st.warning("Username exists. Choose another.")
        return False
    if role == "Checker":
        st.error("Cannot signup as Checker.")
        return False
    st.session_state.users[username] = {"password": hash_password(password), "role": role}
    if role == "Investor":
        st.session_state.users[username]["wallet"] = 10000.0
    st.success("Signup successful. Please login.")
    return True

def login(username: str, password: str):
    u = st.session_state.users.get(username)
    if not u:
        st.error("No such user.")
        return False
    if u["password"] != hash_password(password):
        st.error("Incorrect password.")
        return False
    st.session_state.current_user = {"username": username, "role": u["role"]}
    st.session_state.page = "app"
    st.success(f"Logged in as {username} ({u['role']})")
    st.experimental_rerun()  # reload to switch to app page
    return True

def logout():
    st.session_state.current_user = None
    st.session_state.page = "home"
    st.experimental_rerun()

# ---------- Landing / Cover Page (Sign in / Sign up on screen) ----------
def landing_page():
    # header
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown('<div class="brand"><h1>🚀 CrowdPitch Pro</h1></div>', unsafe_allow_html=True)
        st.markdown('<div class="tag">Verified Crowdfunding Platform — KYC & Manual Verification Demo</div>', unsafe_allow_html=True)
    with col2:
        st.write("")  # reserved for future header items

    st.markdown("---")
    st.markdown('<div class="hero">', unsafe_allow_html=True)

    # centered auth card
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        tab = st.radio("Choose", ["Login", "Signup"], horizontal=True)
        if tab == "Login":
            st.subheader("Sign in")
            li_user = st.text_input("Username", key="li_user")
            li_pass = st.text_input("Password", type="password", key="li_pass")
            if st.button("Login", key="do_login"):
                if li_user and li_pass:
                    login(li_user.strip(), li_pass)
                else:
                    st.warning("Enter username and password.")
            st.markdown('<div class="small-muted">Demo: <b>investor_demo</b>/<i>pass123</i>  or <b>startup_demo</b>/<i>pass123</i></div>', unsafe_allow_html=True)
        else:
            st.subheader("Create account")
            su_user = st.text_input("Choose username", key="su_user")
            su_pass = st.text_input("Choose password", type="password", key="su_pass")
            su_role = st.selectbox("Role", ["Startup", "Investor"], key="su_role")
            if st.button("Create account", key="do_signup"):
                if su_user and su_pass:
                    success = signup(su_user.strip(), su_pass, su_role)
                    if success:
                        # optional: auto-login after signup
                        st.info("You can now login from the login tab.")
                else:
                    st.warning("Provide username and password.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # hero end

    st.markdown("---")
    # Help & FAQs
    help_col1, help_col2 = st.columns([2,1])
    with help_col1:
        st.subheader("How it works")
        st.markdown("""
        1. Startups sign up and submit company info + KYC documents + performance CSV.  
        2. A hidden Checker reviews uploaded docs and approves or rejects the pitch.  
        3. Approved pitches show up for Investors who can view forecasts and invest (simulated wallet).  
        4. Investors can raise complaints which go to the Checker.
        """)
    with help_col2:
        st.subheader("FAQs")
        st.markdown("""
        **Q:** Is this real funding?  
        **A:** No — this is a demo / prototype for onboarding & verification flows.
        
        **Q:** Who is the Checker?  
        **A:** A manual reviewer (hidden account). Use the provided secret credentials for testing.
        
        **Q:** Can I persist data across restarts?  
        **A:** Not yet. We can add SQLite persistence if you want.
        """)

    st.markdown("<small class='small-muted'>Hidden checker credentials (keep secret): <b>checker_agent</b> / <i>Check@2025!</i></small>", unsafe_allow_html=True)

# ---------- Main App (after login) ----------
def main_app():
    user = st.session_state.current_user
    username = user["username"]
    role = user["role"]

    # Top bar with logout & avatar
    top_left, top_right = st.columns([5,1])
    with top_left:
        st.markdown(f"### Welcome — {username} ({role})")
    with top_right:
        if st.button("Logout"):
            logout()
    st.markdown("---")

    # ---------- STARTUP ----------
    if role == "Startup":
        st.header("🏢 Startup Onboarding & Pitch Creation")
        st.markdown("Fill business details, upload KYC/legal docs, upload CSV for forecasting, and submit your pitch. A checker will review documents before publishing.")

        with st.form("onboard_form", clear_on_submit=False):
            st.subheader("Company Identity")
            company_name = st.text_input("Company Legal Name", placeholder="e.g., Acme Pvt Ltd")
            reg_number = st.text_input("Business Registration / Incorporation Number")
            country = st.text_input("Country of Registration", value="India")
            founders = st.text_area("Founders' Full Names & Roles (comma separated)", placeholder="Alice (CEO), Bob (CTO)")
            official_email = st.text_input("Official Email (company domain preferred)")
            website = st.text_input("Company Website (https://...)")
            linkedin = st.text_input("LinkedIn URL (optional)")
            x_link = st.text_input("X / Twitter URL (optional)")
            insta = st.text_input("Instagram URL (optional)")

            st.subheader("Documents & Media")
            logo_file = st.file_uploader("Logo & Brand Asset (PNG/JPG)", type=["png","jpg","jpeg"])
            kyc_file = st.file_uploader("KYC Document (ID) (PNG/PDF)", type=["png","jpg","jpeg","pdf"])
            address_file = st.file_uploader("Proof of Address (PNG/PDF)", type=["png","jpg","jpeg","pdf"])
            bank_file = st.file_uploader("Bank Account Verification Document (PNG/PDF)", type=["png","jpg","jpeg","pdf"])

            st.subheader("Pitch & Data")
            pitch_name = st.text_input("Pitch / Product Name")
            short_desc = st.text_input("Short Description")
            long_desc = st.text_area("Long Description")
            csv_file = st.file_uploader("Performance CSV (date,value) for forecasting", type=["csv"])
            video_link = st.text_input("Video pitch link (any URL)")

            submitted = st.form_submit_button("Submit Pitch & Onboard")

        if submitted:
            missing = []
            if not company_name: missing.append("Company Legal Name")
            if not pitch_name: missing.append("Pitch Name")
            if not csv_file: missing.append("CSV data")
            if not kyc_file: missing.append("KYC Document")
            if not address_file: missing.append("Proof of Address")
            if not bank_file: missing.append("Bank Document")
            if missing:
                st.error("Missing required fields: " + ", ".join(missing))
            else:
                # parse CSV
                try:
                    df = pd.read_csv(csv_file)
                    df.columns = ["date", "value"]
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date")
                except Exception as e:
                    st.error(f"CSV parsing error: {e}")
                    st.stop()

                logo_bytes = None
                if logo_file:
                    try:
                        logo_bytes = image_bytes(logo_file)
                    except:
                        logo_bytes = None

                # heuristics
                website_domain = extract_domain(website) if website else ""
                email_dom = email_domain(official_email) if official_email else ""
                email_ok = False
                if official_email and website_domain:
                    email_ok = (website_domain in email_dom) or (email_dom in website_domain)
                else:
                    if official_email and not any(d in official_email for d in ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]):
                        email_ok = True

                bank_fname = getattr(bank_file, "name", "") if bank_file else ""
                bank_match = name_in_filename(company_name, bank_fname)

                verification = {
                    "email_domain": "Verified" if email_ok else "Pending/Check",
                    "bank_doc": "Verified" if bank_match else "Pending/Manual Review",
                    "kyc_uploaded": True,
                    "address_uploaded": True
                }

                pitch = {
                    "company_name": company_name,
                    "reg_number": reg_number,
                    "country": country,
                    "founders": founders,
                    "official_email": official_email,
                    "website": website,
                    "socials": {"linkedin": linkedin, "x": x_link, "instagram": insta},
                    "logo": logo_bytes,
                    "kyc_file_name": getattr(kyc_file, "name", "") if kyc_file else "",
                    "address_file_name": getattr(address_file, "name", "") if address_file else "",
                    "bank_file_name": bank_fname,
                    "pitch_name": pitch_name,
                    "short": short_desc,
                    "desc": long_desc,
                    "owner": username,
                    "data": df,
                    "video": video_link,
                    "funded": 0.0,
                    "investors": [],
                    "verification": verification,
                    "published": False,
                    "checker_status": "Pending",
                    "checker_note": "",
                    "checker_decision": None
                }
                st.session_state.pitches.append(pitch)
                st.success(f"Pitch '{pitch_name}' created and submitted for checker review.")
                st.balloons()

        st.markdown("---")
        st.subheader("Your Created Pitches")
        mine = [p for p in st.session_state.pitches if p["owner"] == username]
        if not mine:
            st.info("You have no created pitches yet.")
        else:
            for p in mine:
                st.markdown(f"### {p['pitch_name']}")
                cols = st.columns([1,3])
                with cols[0]:
                    if p["logo"]:
                        st.image(p["logo"], width=180)
                    else:
                        st.image("https://via.placeholder.com/220x140.png?text=No+Logo", width=180)
                    st.markdown(f"**Funded:** ₹{p['funded']:,.2f}")
                    st.markdown("**Checker Status**")
                    st.write({
                        "checker_status": p["checker_status"],
                        "published": p["published"],
                        "checker_note": p.get("checker_note","")
                    })
                with cols[1]:
                    st.markdown(p["short"])
                    if st.button(f"View & Forecast — {p['pitch_name']}", key=f"view_{p['pitch_name']}"):
                        st.session_state._view_pitch = p["pitch_name"]

    # ---------- CHECKER ----------
    elif role == "Checker":
        st.header("🕵️ Checker Dashboard — Review & Complaints")
        st.markdown("Approve or reject pitches after reviewing documents and complaints. Add notes as needed.")

        pending = [p for p in st.session_state.pitches if p["checker_status"] == "Pending"]
        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader("Pending Pitches")
            if not pending:
                st.info("No pending pitches to review.")
            else:
                for p in pending:
                    st.markdown(f"### {p['pitch_name']} — {p['company_name']}")
                    c_left, c_right = st.columns([1,2])
                    with c_left:
                        if p["logo"]:
                            st.image(p["logo"], width=160)
                        else:
                            st.image("https://via.placeholder.com/220x140.png?text=No+Logo", width=160)
                    with c_right:
                        st.markdown(p["short"])
                        st.markdown("**Uploaded files (filenames)**")
                        st.write({
                            "KYC": p.get("kyc_file_name",""),
                            "Address": p.get("address_file_name",""),
                            "Bank": p.get("bank_file_name","")
                        })
                        st.markdown("**Verification heuristics**")
                        st.write(p["verification"])
                        st.markdown("**Preview data (top 5 rows)**")
                        st.dataframe(p["data"].head())
                        note = st.text_area("Decision note (optional)", key=f"note_{p['pitch_name']}")
                        cols = st.columns([1,1,1])
                        if cols[0].button("Approve", key=f"approve_{p['pitch_name']}"):
                            p["checker_status"] = "Approved"
                            p["published"] = True
                            p["checker_decision"] = "approved"
                            p["checker_note"] = note or "Approved by checker."
                            st.success(f"Approved {p['pitch_name']}")
                        if cols[1].button("Reject", key=f"reject_{p['pitch_name']}"):
                            p["checker_status"] = "Rejected"
                            p["published"] = False
                            p["checker_decision"] = "rejected"
                            p["checker_note"] = note or "Rejected by checker."
                            st.error(f"Rejected {p['pitch_name']}")
                        if cols[2].button("Mark for Re-check", key=f"recheck_{p['pitch_name']}"):
                            p["checker_status"] = "Pending"
                            p["published"] = False
                            p["checker_decision"] = None
                            p["checker_note"] = note or "Marked for re-check."
                            st.info("Marked for re-check")

        with col2:
            st.subheader("Complaints (All)")
            complaints = st.session_state.complaints
            if not complaints:
                st.info("No complaints yet.")
            else:
                for i, c in enumerate(complaints):
                    st.markdown(f"**#{i+1}** — Pitch: **{c['pitch_name']}** — by {c['investor']}")
                    st.markdown(f"- Status: {c.get('status','Open')}")
                    st.markdown(f"- Message: {c['message']}")
                    st.markdown(f"- Created: {c.get('created','-')}")
                    st.markdown(f"- Resolution note: {c.get('resolution_note','')}")
                    cols = st.columns([1,1])
                    if cols[0].button("Resolve", key=f"resolve_{i}"):
                        c["status"] = "Resolved"
                        st.success("Complaint marked resolved.")
                    if cols[1].button("Add note", key=f"cnote_{i}"):
                        tn = st.text_input("Resolution note", key=f"input_note_{i}")
                        if tn:
                            c["resolution_note"] = tn
                            st.success("Note saved.")

    # ---------- INVESTOR ----------
    elif role == "Investor":
        st.header("💼 Investor Dashboard — Explore & Invest")
        st.markdown("Browse approved startups, view forecasts, watch video, raise complaints, and invest using simulated wallet.")

        published_pitches = [p for p in st.session_state.pitches if p.get("published", False)]
        if not published_pitches:
            st.info("No published pitches yet. Wait for startups to be approved by the checker.")
            st.stop()

        # Filters & search
        q_col, s_col = st.columns([3,1])
        query = q_col.text_input("Search by name or description")
        sort_by = s_col.selectbox("Sort by", ["Newest", "Most Funded", "Verified First"])
        pitches = published_pitches.copy()

        if query:
            pitches = [p for p in pitches if query.lower() in (p.get("pitch_name","").lower() + p.get("company_name","").lower() + p.get("short","").lower() + p.get("desc","").lower())]

        if sort_by == "Most Funded":
            pitches = sorted(pitches, key=lambda x: x["funded"], reverse=True)
        elif sort_by == "Verified First":
            pitches = sorted(pitches, key=lambda x: (x["verification"].get("email_domain")!="Verified", x["verification"].get("bank_doc")!="Verified"))
        else:
            pitches = list(reversed(pitches))

        for p in pitches:
            st.markdown("", unsafe_allow_html=True)
            card_cols = st.columns([1.2, 3])
            with card_cols[0]:
                if p["logo"]:
                    st.image(p["logo"], width=180)
                else:
                    st.image("https://via.placeholder.com/220x140.png?text=No+Logo", width=180)
                st.markdown(f"**Funded:** ₹{p['funded']:,.2f}")
                st.markdown("**Verification**")
                st.write(p["verification"])
            with card_cols[1]:
                st.markdown(f"### {p['pitch_name']} ({p['company_name']})")
                st.markdown(f"_{p['short']}_")
                st.write(p["desc"][:350] + ("..." if len(p["desc"])>350 else ""))
                df = p["data"]
                latest = df['value'].iloc[-1]
                meanv = df['value'].mean()
                growth = (df['value'].iloc[-1] / df['value'].iloc[0] - 1)*100
                cols = st.columns(4)
                cols[0].metric("Latest", f"{latest:.2f}")
                cols[1].metric("Average", f"{meanv:.2f}")
                cols[2].metric("Growth (%)", f"{growth:.1f}%")
                cols[3].metric("Funded", f"₹{p['funded']:,.0f}")

                b1, b2, b3 = st.columns([1,1,1])
                if b1.button("View Details", key=f"view_{p['pitch_name']}"):
                    st.session_state._view_pitch = p["pitch_name"]
                if b2.button("Invest", key=f"invest_{p['pitch_name']}"):
                    st.session_state._invest_in = p["pitch_name"]
                if b3.button("Complaint", key=f"complain_{p['pitch_name']}"):
                    st.session_state._complaint_for = p["pitch_name"]
            st.markdown("---")

        # View details
        if "_view_pitch" in st.session_state:
            target = st.session_state._view_pitch
            selected = next((x for x in st.session_state.pitches if x["pitch_name"] == target), None)
            if selected:
                st.markdown(f"## 🔎 {selected['pitch_name']} — {selected['company_name']}")
                left, right = st.columns([1,2])
                with left:
                    if selected["logo"]:
                        st.image(selected["logo"], use_column_width=True)
                    else:
                        st.image("https://via.placeholder.com/320x180.png?text=No+Logo", use_column_width=True)
                    st.markdown("**Owner:** " + selected["owner"])
                    st.markdown("**Registration:** " + (selected["reg_number"] or "—"))
                    st.markdown("**Country:** " + (selected["country"] or "—"))
                    st.markdown("**Verification summary**")
                    st.write(selected["verification"])
                    if selected["kyc_file_name"]:
                        st.markdown("KYC doc: " + selected["kyc_file_name"])
                    if selected["address_file_name"]:
                        st.markdown("Address proof: " + selected["address_file_name"])
                    if selected["bank_file_name"]:
                        st.markdown("Bank doc: " + selected["bank_file_name"])
                    if selected["video"]:
                        st.markdown("🎥 Video pitch")
                        st.video(selected["video"])
                with right:
                    st.markdown(selected["desc"])
                    st.markdown("### Historical data & Forecast")
                    df = selected["data"]

                    fig, ax = plt.subplots(figsize=(9,4))
                    sns.lineplot(x="date", y="value", data=df, marker="o", ax=ax, label="Historical")
                    forecast_available = False
                    try:
                        model = sm.tsa.ARIMA(df["value"], order=(1,1,1))
                        result = model.fit()
                        n = 6
                        forecast = result.forecast(steps=n)
                        conf = result.get_forecast(steps=n).conf_int()
                        future_dates = pd.date_range(df["date"].iloc[-1], periods=n+1, freq="M")[1:]
                        ax.plot(future_dates, forecast, marker="X", linestyle="--", label="Forecast")
                        ax.fill_between(future_dates, conf.iloc[:,0], conf.iloc[:,1], alpha=0.2)
                        forecast_available = True
                    except Exception as e:
                        st.warning("Forecast error: " + str(e))
                    ax.set_title("Metric (historical + forecast)")
                    st.pyplot(fig)

                    if forecast_available:
                        fdf = pd.DataFrame({
                            "date": future_dates,
                            "forecast": forecast,
                            "lower": conf.iloc[:,0],
                            "upper": conf.iloc[:,1]
                        })
                        st.download_button("Download forecast CSV", df_to_csv_bytes(fdf), file_name=f"{selected['pitch_name']}_forecast.csv")

                if st.button("Close view"):
                    del st.session_state._view_pitch

        # Complaint flow
        if "_complaint_for" in st.session_state:
            pitch_name = st.session_state._complaint_for
            st.markdown(f"## ❗ Raise Complaint for {pitch_name}")
            msg = st.text_area("Describe the issue (why is this fake / what happened?)")
            if st.button("Submit Complaint"):
                if not msg or len(msg.strip()) < 10:
                    st.warning("Please provide a clearer complaint (min 10 chars).")
                else:
                    st.session_state.complaints.append({
                        "pitch_name": pitch_name,
                        "investor": username,
                        "message": msg.strip(),
                        "status": "Open",
                        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "resolution_note": ""
                    })
                    st.success("Complaint submitted to checker.")
                    del st.session_state._complaint_for

        # Invest flow
        if "_invest_in" in st.session_state:
            target = st.session_state._invest_in
            selected = next((x for x in st.session_state.pitches if x["pitch_name"] == target), None)
            if selected:
                st.markdown(f"## 💸 Invest in {selected['pitch_name']}")
                st.write(selected["short"])
                amount = st.number_input("Amount (INR)", min_value=100.0, value=500.0, step=100.0)
                wallet = st.session_state.users[username]["wallet"]
                st.markdown(f"**Your wallet:** ₹{wallet:,.2f}")
                if st.button("Confirm Invest"):
                    if amount <= 0:
                        st.error("Enter positive amount")
                    elif amount > wallet:
                        st.error("Insufficient balance")
                    else:
                        st.session_state.users[username]["wallet"] -= amount
                        selected["funded"] += amount
                        selected["investors"].append({"investor": username, "amount": amount})
                        st.session_state.investments.append({"investor": username, "pitch": selected["pitch_name"], "amount": amount})
                        st.success(f"Invested ₹{amount:,.0f} in {selected['pitch_name']}")
                        del st.session_state._invest_in

    # Footer stats (global)
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Pitches", f"{len(st.session_state.pitches)}")
    with c2:
        total_funded = sum(p["funded"] for p in st.session_state.pitches)
        st.metric("Total Funded", f"₹{total_funded:,.0f}")
    with c3:
        st.metric("Total Complaints", f"{len(st.session_state.complaints)}")

    st.markdown("<center>Built with ❤️ — CrowdPitch Pro Demo</center>", unsafe_allow_html=True)

# ---------- Router ----------
if st.session_state.page == "home" or st.session_state.current_user is None:
    landing_page()
else:
    main_app()
