# app.py
"""
CrowdPitch Pro — KYC Edition (Enhanced)
- Landing page (login/signup)
- Startup onboarding (more docs + target funding)
- Checker with inline PDF/image preview
- Investor marketplace with search/filters + card grid
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

# ---------- CSS for cards/grid ----------
st.markdown(
    """
    <style>
    body { background: linear-gradient(180deg,#071A2A,#071622); color: #e6eef6; }
    .auth-card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); padding:18px; border-radius:10px; box-shadow:0 6px 18px rgba(0,0,0,0.45); }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); border-radius:12px; padding:12px; box-shadow:0 6px 18px rgba(0,0,0,0.45); margin-bottom:14px;}
    .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap:16px; }
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
    if not name or not fname: return False
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

def safe_read_csv(file) -> pd.DataFrame:
    # read and canonicalize
    df = pd.read_csv(file)
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df

def predict_forecast_and_roi(df: pd.DataFrame, steps: int = 6):
    # try ARIMA, fallback to linear projection
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
        # fallback simple projection
        try:
            if len(df) < 3:
                return None, None, None, 0.0
            pct_changes = df["value"].pct_change().dropna()
            avg = pct_changes[-3:].mean() if len(pct_changes) >= 3 else pct_changes.mean()
            latest = df["value"].iloc[-1]
            vals = []
            dates = pd.date_range(df["date"].iloc[-1], periods=steps+1, freq="M")[1:]
            v = latest
            for _ in range(steps):
                v = v * (1 + avg)
                vals.append(v)
            mean_forecast = np.mean(vals)
            roi = (mean_forecast / latest - 1) * 100.0
            return dates, np.array(vals), None, roi
        except Exception:
            return None, None, None, 0.0

def embed_pdf_bytes(pdf_bytes: bytes, width="100%", height="600px"):
    # embed pdf using base64 data URI (works for moderate sizes)
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    src = f"data:application/pdf;base64,{b64}"
    html_tag = f'<iframe src="{src}" width="{width}" height="{height}"></iframe>'
    st.components.v1.html(html_tag, height=int(height.replace("px",""))+20)

def embed_image_bytes(img_bytes: bytes, width=300):
    st.image(img_bytes, width=width)

# ---------- session init ----------
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

# demo & checker
if "investor_demo" not in st.session_state.users:
    st.session_state.users["investor_demo"] = {"password": hash_password("pass123"), "role": "Investor", "wallet": 10000.0}
if "startup_demo" not in st.session_state.users:
    st.session_state.users["startup_demo"] = {"password": hash_password("pass123"), "role": "Startup"}
if "checker_agent" not in st.session_state.users:
    st.session_state.users["checker_agent"] = {"password": hash_password("Check@2025!"), "role": "Checker"}

# helper ui keys
if "_view_graph" not in st.session_state: st.session_state._view_graph = None
if "_complaint_for" not in st.session_state: st.session_state._complaint_for = None
if "_checker_view" not in st.session_state: st.session_state._checker_view = None

# ---------- auth ----------
def signup(username, password, role):
    if not username or not password:
        st.warning("Enter username & password.")
        return False
    if username in st.session_state.users:
        st.warning("Username taken.")
        return False
    if role == "Checker":
        st.error("Cannot signup as Checker.")
        return False
    st.session_state.users[username] = {"password": hash_password(password), "role": role}
    if role == "Investor":
        st.session_state.users[username]["wallet"] = 10000.0
    st.success("Account created — please login.")
    return True

def login(username, password):
    u = st.session_state.users.get(username)
    if not u:
        st.error("No such user.")
        return False
    if u["password"] != hash_password(password):
        st.error("Incorrect password.")
        return False
    st.session_state.current_user = {"username": username, "role": u["role"]}
    st.session_state.page = "app"
    st.success(f"Welcome {username} ({u['role']})")
    st.rerun()
    return True

def logout():
    st.session_state.current_user = None
    st.session_state.page = "home"
    st.rerun()

# ---------- landing / cover ----------
def landing_page():
    st.markdown("<h1 style='text-align:center;color:#06b6d4;'>🚀 CrowdPitch Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#9fb4c9;'>Verified Crowdfunding Platform — KYC & Manual Verification Demo</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        mode = st.radio("Action", ["Login", "Signup"], horizontal=True)
        if mode == "Login":
            st.subheader("Sign in")
            li_user = st.text_input("Username", key="li_user")
            li_pass = st.text_input("Password", type="password", key="li_pass")
            if st.button("Login"):
                if li_user and li_pass:
                    login(li_user.strip(), li_pass)
                else:
                    st.warning("Enter username and password.")
            st.markdown("<div class='small-muted'>Demo accounts: <b>investor_demo</b>/<i>pass123</i> or <b>startup_demo</b>/<i>pass123</i></div>", unsafe_allow_html=True)
        else:
            st.subheader("Create account")
            su_user = st.text_input("Choose username", key="su_user")
            su_pass = st.text_input("Choose password", type="password", key="su_pass")
            role = st.selectbox("Role", ["Startup", "Investor"], key="su_role")
            if st.button("Create account"):
                if su_user and su_pass:
                    signup(su_user.strip(), su_pass, role)
                else:
                    st.warning("Enter username & password.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    left, right = st.columns([2,1])
    with left:
        st.subheader("About CrowdPitch Pro")
        st.write(
            "CrowdPitch Pro is a demo platform to showcase startup onboarding with multi-document KYC, "
            "manual checker review, investor marketplace with predictions, and complaint workflows."
        )
        st.markdown("**Contact / Support**")
        st.write("Email: support@crowdpitch.example")
        st.write("Twitter: @crowdpitch")
    with right:
        st.subheader("FAQs")
        st.write("Is this real funding? — No, demo only.")
        st.write("Checker (hidden): `checker_agent` / `Check@2025!`")
    st.markdown("---")

# ---------- startup page (with target funding + store file bytes) ----------
def startup_page(user):
    st.header("🏢 Startup Onboarding")
    st.markdown("Provide company details, upload required documents and media, set a funding target, and upload performance CSV for ROI prediction.")

    with st.form("onboard_form", clear_on_submit=False):
        st.subheader("Company Info")
        company_name = st.text_input("Company Legal Name")
        reg_number = st.text_input("Business Registration / Incorporation Number")
        country = st.text_input("Country", value="India")
        founders = st.text_area("Founders (comma separated)")
        official_email = st.text_input("Official Email")
        website = st.text_input("Website (https://...)")

        st.subheader("Media")
        logo_file = st.file_uploader("Logo (PNG/JPG)", type=["png","jpg","jpeg"])
        company_img_file = st.file_uploader("Company Image (team/office)", type=["png","jpg","jpeg"])
        video_link = st.text_input("Video pitch link (YouTube/Vimeo)")

        st.subheader("Required Documents")
        kyc_file = st.file_uploader("KYC (ID) (PNG/PDF)", type=["png","jpg","jpeg","pdf"])
        address_file = st.file_uploader("Address proof (PNG/PDF)", type=["png","jpg","jpeg","pdf"])
        bank_file = st.file_uploader("Bank verification (PNG/PDF)", type=["png","jpg","jpeg","pdf"])
        tax_file = st.file_uploader("Tax doc (optional) (PNG/PDF)", type=["png","jpg","jpeg","pdf"])
        inc_file = st.file_uploader("Incorporation cert (optional) (PNG/PDF)", type=["png","jpg","jpeg","pdf"])

        st.subheader("Pitch & Funding")
        pitch_name = st.text_input("Pitch / Product Name")
        short_desc = st.text_input("Short Description (1-2 lines)")
        long_desc = st.text_area("Long Description")
        target_funding = st.number_input("Target Funding (INR)", min_value=0.0, value=100000.0, step=1000.0)
        csv_file = st.file_uploader("Performance CSV (date,value)", type=["csv"])

        submitted = st.form_submit_button("Submit Pitch")

    if submitted:
        missing = []
        if not company_name: missing.append("Company Name")
        if not pitch_name: missing.append("Pitch Name")
        if not csv_file: missing.append("CSV data")
        if not kyc_file: missing.append("KYC")
        if not address_file: missing.append("Address")
        if not bank_file: missing.append("Bank doc")
        if missing:
            st.error("Missing required: " + ", ".join(missing))
            return

        try:
            df = safe_read_csv(csv_file)
        except Exception as e:
            st.error(f"CSV parse error: {e}")
            return

        # read bytes for uploaded files so checker can preview
        def read_bytes(f):
            try:
                return f.read()
            except:
                return None

        logo_b = None
        try:
            if logo_file:
                logo_b = image_bytes(logo_file)
        except:
            logo_b = None

        comp_img_b = None
        try:
            if company_img_file:
                comp_img_b = image_bytes(company_img_file)
        except:
            comp_img_b = None

        kyc_b = read_bytes(kyc_file) if kyc_file else None
        addr_b = read_bytes(address_file) if address_file else None
        bank_b = read_bytes(bank_file) if bank_file else None
        tax_b = read_bytes(tax_file) if tax_file else None
        inc_b = read_bytes(inc_file) if inc_file else None

        pitch = {
            "company_name": company_name,
            "reg_number": reg_number,
            "country": country,
            "founders": founders,
            "official_email": official_email,
            "website": website,
            "logo": logo_b,
            "company_image": comp_img_b,
            "video": video_link,
            "kyc_bytes": kyc_b,
            "kyc_file_name": getattr(kyc_file, "name", ""),
            "address_bytes": addr_b,
            "address_file_name": getattr(address_file, "name", ""),
            "bank_bytes": bank_b,
            "bank_file_name": getattr(bank_file, "name", ""),
            "tax_bytes": tax_b,
            "tax_file_name": getattr(tax_file, "name", ""),
            "inc_bytes": inc_b,
            "inc_file_name": getattr(inc_file, "name", ""),
            "pitch_name": pitch_name,
            "short": short_desc,
            "desc": long_desc,
            "owner": user["username"],
            "data": df,
            "target": float(target_funding),
            "funded": 0.0,
            "investors": [],
            "verification": {"kyc": True, "address": True, "bank": True},
            "published": False,
            "checker_status": "Pending",
            "checker_note": ""
        }

        st.session_state.pitches.append(pitch)
        st.success(f"Pitch '{pitch_name}' submitted for checker review.")
        st.balloons()

    st.markdown("---")
    st.subheader("Your pitches")
    mine = [p for p in st.session_state.pitches if p["owner"] == user["username"]]
    if not mine:
        st.info("No pitches yet.")
    for p in mine:
        cols = st.columns([1,3])
        with cols[0]:
            if p.get("logo"):
                st.image(p["logo"], width=160)
            else:
                st.image("https://via.placeholder.com/220x140.png?text=No+Logo", width=160)
            st.markdown(f"**Funded:** ₹{p['funded']:,.0f}")
            st.markdown(f"**Target:** ₹{p['target']:,.0f}")
            st.markdown(f"Status: {p['checker_status']}")
        with cols[1]:
            st.markdown(f"### {p['pitch_name']}")
            st.write(p.get("short",""))
            if st.button(f"View (owner) — {p['pitch_name']}", key=f"owner_view_{p['pitch_name']}"):
                st.session_state._view_graph = p["pitch_name"]

# ---------- checker page with document preview ----------
def checker_page(user):
    st.header("🕵️ Checker Dashboard")
    tabs = st.tabs(["Pending", "Approved", "Rejected", "Complaints"])
    with tabs[0]:
        pending = [p for p in st.session_state.pitches if p["checker_status"] == "Pending"]
        if not pending:
            st.info("No pending pitches.")
        for p in pending:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            r1, r2 = st.columns([1,3])
            with r1:
                if p.get("logo"):
                    st.image(p["logo"], width=140)
                else:
                    st.image("https://via.placeholder.com/220x140.png?text=No+Logo", width=140)
                st.markdown(f"Owner: {p['owner']}")
                st.markdown(f"{status_badge_html(p['checker_status'])}", unsafe_allow_html=True)
            with r2:
                st.markdown(f"### {p['pitch_name']} — {p['company_name']}")
                st.write(p.get("short",""))
                if st.button("View Details", key=f"view_{p['pitch_name']}"):
                    st.session_state._checker_view = p["pitch_name"]
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        approved = [p for p in st.session_state.pitches if p["checker_status"] == "Approved"]
        if not approved:
            st.info("No approved pitches.")
        for p in approved:
            st.markdown(f"- {p['pitch_name']} ({p['company_name']})")

    with tabs[2]:
        rejected = [p for p in st.session_state.pitches if p["checker_status"] == "Rejected"]
        if not rejected:
            st.info("No rejected pitches.")
        for p in rejected:
            st.markdown(f"- {p['pitch_name']} ({p['company_name']})")

    with tabs[3]:
        comps = st.session_state.complaints
        if not comps:
            st.info("No complaints.")
        else:
            for i,c in enumerate(comps):
                st.markdown(f"**#{i+1}** Pitch: {c['pitch_name']} — by {c['investor']}")
                st.markdown(f"- Message: {c['message']}")
                st.markdown(f"- Status: {c.get('status','Open')}")
                if st.button("Mark Resolved", key=f"res_{i}"):
                    c["status"] = "Resolved"
                    st.success("Marked resolved.")

    # Checker details modal / view
    if st.session_state._checker_view:
        target = st.session_state._checker_view
        sel = next((x for x in st.session_state.pitches if x["pitch_name"] == target), None)
        if sel:
            st.markdown("---")
            st.subheader(f"Review: {sel['pitch_name']} — {sel['company_name']}")
            st.write({k: sel.get(k) for k in ["company_name","reg_number","country","founders","official_email","website"]})
            st.markdown("**Media**")
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                st.markdown("Logo")
                if sel.get("logo"):
                    st.image(sel["logo"], width=220)
                else:
                    st.write("—")
            with c2:
                st.markdown("Company Image")
                if sel.get("company_image"):
                    st.image(sel["company_image"], width=220)
                else:
                    st.write("—")
            with c3:
                st.markdown("Video")
                if sel.get("video"):
                    try:
                        st.video(sel.get("video"))
                    except:
                        st.write(html.escape(sel.get("video")))
                else:
                    st.write("—")
            st.markdown("**Documents (click to preview)**")
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown("KYC")
                if sel.get("kyc_bytes"):
                    if st.button("View KYC", key=f"view_kyc_{sel['pitch_name']}"):
                        embed_pdf_bytes(sel["kyc_bytes"], height="600px")
                else:
                    st.write("—")
            with d2:
                st.markdown("Address")
                if sel.get("address_bytes"):
                    if st.button("View Address", key=f"view_addr_{sel['pitch_name']}"):
                        embed_pdf_bytes(sel["address_bytes"], height="600px")
                else:
                    st.write("—")
            with d3:
                st.markdown("Bank")
                if sel.get("bank_bytes"):
                    if st.button("View Bank", key=f"view_bank_{sel['pitch_name']}"):
                        embed_pdf_bytes(sel["bank_bytes"], height="600px")
                else:
                    st.write("—")

            st.markdown("**CSV preview**")
            st.dataframe(sel["data"].head())

            note = st.text_area("Decision note", key=f"note_{sel['pitch_name']}")
            a,b,c = st.columns(3)
            if a.button("Approve", key=f"approve_{sel['pitch_name']}"):
                sel["checker_status"] = "Approved"
                sel["published"] = True
                sel["checker_note"] = note or "Approved"
                st.success("Approved — visible to investors.")
                st.session_state._checker_view = None
            if b.button("Reject", key=f"reject_{sel['pitch_name']}"):
                sel["checker_status"] = "Rejected"
                sel["published"] = False
                sel["checker_note"] = note or "Rejected"
                st.error("Rejected.")
                st.session_state._checker_view = None
            if c.button("Request Re-check", key=f"recheck_{sel['pitch_name']}"):
                sel["checker_status"] = "Pending"
                sel["published"] = False
                sel["checker_note"] = note or "Request re-check"
                st.info("Marked for re-check.")
                st.session_state._checker_view = None

# ---------- investor marketplace + dashboard ----------
def investor_page(user):
    st.header("💼 Investor Marketplace")
    # top controls: wallet + search + filters + toggle to My Investments
    wallet = st.session_state.users[user["username"]].get("wallet", 0.0)
    top1, top2, top3 = st.columns([2,3,1])
    with top1:
        st.markdown(f"**Wallet:** ₹{wallet:,.2f}")
    with top2:
        search = st.text_input("Search by company or pitch")
        # ROI filter slider
        roi_min, roi_max = st.slider("Predicted ROI range (%)", -50, 200, (-10, 50))
    with top3:
        view_mode = st.selectbox("View", ["Marketplace", "My Investments"], key="inv_view_mode")

    if view_mode == "My Investments":
        st.subheader("📊 My Investments")
        invs = [i for i in st.session_state.investments if i["investor"] == user["username"]]
        if not invs:
            st.info("You have no investments yet.")
        else:
            df = pd.DataFrame(invs)
            df["date"] = pd.to_datetime(df["date"])
            total = df["amount"].sum()
            avg = df["predicted_roi"].mean() if "predicted_roi" in df.columns else 0.0
            st.metric("Total invested", f"₹{total:,.2f}")
            st.metric("Average predicted ROI (%)", f"{avg:.1f}%")
            st.dataframe(df.sort_values("date", ascending=False))
        return

    # Marketplace mode
    published = [p for p in st.session_state.pitches if p.get("published")]
    if not published:
        st.info("No published pitches yet.")
        return

    # precompute ROI for each pitch and filter
    listing = []
    for p in published:
        try:
            _, _, _, roi = predict_forecast_and_roi(p["data"], steps=6)
            roi = float(roi) if roi is not None else 0.0
        except:
            roi = 0.0
        p["_predicted_roi"] = roi
        listing.append(p)

    # apply search & ROI filters
    def matches(p):
        if search:
            s = search.lower()
            if s not in (p.get("pitch_name","").lower() + p.get("company_name","").lower() + p.get("short","").lower()):
                return False
        if p["_predicted_roi"] < roi_min or p["_predicted_roi"] > roi_max:
            return False
        return True

    filtered = [p for p in listing if matches(p)]
    st.markdown(f"**{len(filtered)}** results")
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    for p in filtered:
        # compute progress
        target = p.get("target", 0.0) or 0.0
        funded = p.get("funded", 0.0) or 0.0
        pct = (funded / target * 100.0) if target > 0 else 0.0
        roi_display = p.get("_predicted_roi", 0.0)
        # card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        c1, c2 = st.columns([1,2])
        with c1:
            if p.get("logo"):
                st.image(p["logo"], width=150)
            else:
                st.image("https://via.placeholder.com/220x140.png?text=No+Logo", width=150)
        with c2:
            st.markdown(f"### {p['pitch_name']} — {p['company_name']}")
            st.markdown(p.get("short",""))
            st.markdown(f"**Predicted ROI (6mo avg):** {roi_display:.1f}%")
            st.markdown(f"**Funded:** ₹{funded:,.0f} of ₹{target:,.0f}")
            # progress bar
            st.markdown('<div class="progress"><div style="width: {}%"></div></div>'.format(min(max(pct,0),100)), unsafe_allow_html=True)
            # buttons
            b1, b2, b3 = st.columns([1,1,1])
            if b1.button("View Details", key=f"vd_{p['pitch_name']}"):
                st.session_state._view_graph = p["pitch_name"]  # reuse viewer for detail modal
            if b2.button("Invest", key=f"inv_{p['pitch_name']}"):
                # open investment mini-form via session state
                st.session_state["_invest_in"] = p["pitch_name"]
                st.experimental_rerun()
            if b3.button("Complain", key=f"compl_{p['pitch_name']}"):
                st.session_state._complaint_for = p["pitch_name"]
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # investment modal (if set)
    if st.session_state.get("_invest_in"):
        target_name = st.session_state["_invest_in"]
        sel = next((x for x in st.session_state.pitches if x["pitch_name"] == target_name), None)
        if sel:
            st.markdown("---")
            st.subheader(f"Invest in {sel['pitch_name']} — {sel['company_name']}")
            st.write(sel.get("short",""))
            amt = st.number_input("Amount (INR)", min_value=100.0, value=500.0, step=100.0, key=f"amt_invest_{sel['pitch_name']}")
            if st.button("Confirm Invest"):
                wallet = st.session_state.users[user["username"]].get("wallet", 0.0)
                if amt <= 0:
                    st.error("Enter positive amount.")
                elif amt > wallet:
                    st.error("Insufficient balance.")
                else:
                    # commit
                    st.session_state.users[user["username"]]["wallet"] -= amt
                    sel["funded"] += amt
                    sel["investors"].append({"investor": user["username"], "amount": amt, "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                    # record investment with predicted ROI snapshot
                    inv_record = {"investor": user["username"], "pitch": sel["pitch_name"], "amount": amt, "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    try:
                        _, _, _, roi_snap = predict_forecast_and_roi(sel["data"], steps=6)
                        inv_record["predicted_roi"] = float(roi_snap)
                    except:
                        inv_record["predicted_roi"] = 0.0
                    st.session_state.investments.append(inv_record)
                    st.success(f"Invested ₹{amt:,.0f} in {sel['pitch_name']}")
                    st.session_state["_invest_in"] = None
                    st.experimental_rerun()

    # view details (graph + more)
    if st.session_state.get("_view_graph"):
        t = st.session_state["_view_graph"]
        s = next((x for x in st.session_state.pitches if x["pitch_name"] == t), None)
        if s:
            st.markdown("---")
            st.subheader(f"{s['pitch_name']} — {s['company_name']}")
            st.write(s.get("desc",""))
            df = s["data"]
            fig, ax = plt.subplots(figsize=(9,4))
            sns.lineplot(x="date", y="value", data=df, marker="o", ax=ax, label="Historical")
            try:
                fdates, fvals, conf, roi = predict_forecast_and_roi(df, steps=6)
                if fdates is not None:
                    ax.plot(fdates, fvals, marker="X", linestyle="--", label="Forecast")
                    if conf is not None:
                        ax.fill_between(fdates, conf.iloc[:,0], conf.iloc[:,1], alpha=0.2)
            except Exception as e:
                st.warning(f"Forecast error: {e}")
            ax.set_title("Historical + Forecast")
            st.pyplot(fig)
            if st.button("Close"):
                st.session_state._view_graph = None

    # complaint flow
    if st.session_state.get("_complaint_for"):
        pitch_name = st.session_state["_complaint_for"]
        st.markdown("---")
        st.subheader(f"Raise Complaint for {pitch_name}")
        msg = st.text_area("Describe the issue (min 10 chars)")
        if st.button("Submit Complaint"):
            if not msg or len(msg.strip()) < 10:
                st.warning("Please write a clearer complaint (min 10 chars).")
            else:
                st.session_state.complaints.append({
                    "pitch_name": pitch_name,
                    "investor": user["username"],
                    "message": msg.strip(),
                    "status": "Open",
                    "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "resolution_note": ""
                })
                st.success("Complaint submitted.")
                st.session_state._complaint_for = None

# ---------- main app routing ----------
def main_app():
    user = st.session_state.current_user
    st.sidebar.markdown(f"**Logged in as:** {user['username']} ({user['role']})")
    if user["role"] == "Investor":
        st.sidebar.metric("Wallet (₹)", f"{st.session_state.users[user['username']].get('wallet',0.0):,.2f}")
    if st.sidebar.button("Logout"):
        logout()

    if user["role"] == "Startup":
        startup_page(user)
    elif user["role"] == "Checker":
        checker_page(user)
    elif user["role"] == "Investor":
        investor_page(user)
    else:
        st.info("Unknown role.")

# ---------- router ----------
if st.session_state.page == "home" or st.session_state.current_user is None:
    landing_page()
else:
    main_app()
