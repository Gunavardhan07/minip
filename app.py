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

st.set_page_config(page_title="SeedConnect — KYC Edition", page_icon="🚀", layout="wide")
sns.set_style("darkgrid")
plt.rcParams["figure.dpi"] = 100

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
    .product-card { padding:12px; border-radius:8px; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); box-shadow:0 6px 18px rgba(0,0,0,0.45); }
    .cart { position: sticky; top: 20px; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    src = f"data:application/pdf;base64,{b64}"
    html_tag = f'<iframe src="{src}" width="{width}" height="{height}"></iframe>'
    st.components.v1.html(html_tag, height=int(height.replace("px",""))+20)

def embed_image_bytes(img_bytes: bytes, width=300):
    st.image(img_bytes, width=width)

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
if "cart" not in st.session_state:
    st.session_state.cart = []
if "orders" not in st.session_state:
    st.session_state.orders = []

if "investor_seedconnect" not in st.session_state.users:
    st.session_state.users["investor_seedconnect"] = {"password": hash_password("pass123"), "role": "Investor", "wallet": 10000.0}
if "startup_seedconnect" not in st.session_state.users:
    st.session_state.users["startup_seedconnect"] = {"password": hash_password("pass123"), "role": "Startup"}
if "compliance_officer" not in st.session_state.users:
    st.session_state.users["compliance_officer"] = {"password": hash_password("Check@2025!"), "role": "Checker"}

if "_view_graph" not in st.session_state:
    st.session_state._view_graph = None
if "_complaint_for" not in st.session_state:
    st.session_state._complaint_for = None
if "_checker_view" not in st.session_state:
    st.session_state._checker_view = None

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

def landing_page():
    # --- HERO HEADER ---
    st.markdown("""
        <div style='text-align:center;padding:60px 20px;background:linear-gradient(90deg,#071A2A,#09344E);border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,0.4);'>
            <h1 style='color:#06b6d4;font-size:3rem;margin-bottom:10px;'>🚀 SeedConnect</h1>
            <h3 style='color:#e6eef6;margin-top:0;'>Empowering Startups, Simplifying Investments.</h3>
            <p style='color:#9fb4c9;font-size:1rem;margin-top:10px;'>
                A verified, secure, and transparent crowdfunding platform built for entrepreneurs and investors — powered by real-time KYC and AI-driven insights.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- AUTH CARD (Login / Signup) ---
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        mode = st.radio("Choose an action", ["Login", "Signup"], horizontal=True)
        if mode == "Login":
            st.subheader("Sign In to Your Account")
            li_user = st.text_input("Username", key="li_user")
            li_pass = st.text_input("Password", type="password", key="li_pass")
            if st.button("Login"):
                if li_user and li_pass:
                    login(li_user.strip(), li_pass)
                else:
                    st.warning("Enter both username and password.")
        else:
            st.subheader("Create a New Account")
            su_user = st.text_input("Choose username", key="su_user")
            su_pass = st.text_input("Choose password", type="password", key="su_pass")
            role = st.selectbox("Role", ["Startup", "Investor"], key="su_role")
            if st.button("Create account"):
                if su_user and su_pass:
                    signup(su_user.strip(), su_pass, role)
                else:
                    st.warning("Enter username & password.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # --- HOW IT WORKS ---
    st.markdown("""
        <h2 style='text-align:center;color:#06b6d4;'>🌱 How It Works</h2>
        <p style='text-align:center;color:#9fb4c9;'>Three simple steps to connect Startups, Investors, and Compliance Officers in one secure ecosystem.</p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class='card'>
                <h4 style='color:#06b6d4;'>1️⃣ For Startups</h4>
                <ul style='color:#c9d7e8;'>
                    <li>Register and complete KYC verification.</li>
                    <li>Upload key company documents and logo.</li>
                    <li>Submit financials for ROI prediction.</li>
                    <li>Wait for compliance review & approval.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class='card'>
                <h4 style='color:#06b6d4;'>2️⃣ For Investors</h4>
                <ul style='color:#c9d7e8;'>
                    <li>Sign up and load wallet securely.</li>
                    <li>Browse verified startup listings.</li>
                    <li>View predicted ROI forecasts.</li>
                    <li>Invest safely and track your portfolio.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class='card'>
                <h4 style='color:#06b6d4;'>3️⃣ For Compliance Officers</h4>
                <ul style='color:#c9d7e8;'>
                    <li>Manually review uploaded KYC docs.</li>
                    <li>Approve or reject startup listings.</li>
                    <li>Ensure platform integrity and trust.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # --- FAQs SECTION ---
    st.markdown("<h2 style='text-align:center;color:#06b6d4;'>❓ Frequently Asked Questions</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#9fb4c9;'>Everything you need to know before getting started.</p>", unsafe_allow_html=True)

    faqs = [
        ("Is SeedConnect a real investment platform?", 
         "Currently, this is a demo environment designed for educational and testing purposes."),
        ("What kind of startups can apply?", 
         "Any legally registered startup or SME with valid business documents can apply."),
        ("What documents are required for KYC?", 
         "Company registration, PAN, address proof, director ID, and financial statements."),
        ("How are investors protected?", 
         "All startups undergo a compliance review, and only approved listings can receive investments."),
        ("Can I withdraw funds from my wallet?", 
         "In this demo version, wallet funds are simulated and not withdrawable."),
        ("How is ROI calculated?", 
         "SeedConnect uses ARIMA-based time-series forecasting to estimate short-term ROI trends."),
        ("Is my data secure?", 
         "All uploads are processed securely, and sensitive data is never shared publicly."),
        ("What happens if my application is rejected?", 
         "You can reapply after addressing the compliance officer’s feedback."),
        ("Is there any fee for joining?", 
         "No, creating an account as a startup or investor is completely free."),
        ("How do I contact support?", 
         "You can reach us at <b>support@seedconnect.com</b> or via Twitter <b>@seedconnect</b>."),
    ]

    for q, a in faqs:
        with st.expander(q):
            st.markdown(f"<p style='color:#dbe7f3;'>{a}</p>", unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # --- CONTACT SECTION ---
    st.markdown("""
        <div style='background:linear-gradient(180deg,#0b2235,#071A2A);padding:40px;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.4);'>
            <h2 style='color:#06b6d4;text-align:center;'>📬 Contact & Company Information</h2>
            <div style='display:flex;justify-content:space-around;flex-wrap:wrap;margin-top:20px;color:#bcd1e3;'>
                <div style='max-width:300px;'>
                    <h4>🏢 Headquarters</h4>
                    <p>SeedConnect Technologies Pvt. Ltd.<br>
                    21st Floor, Orion Business Hub,<br>
                    Bengaluru, India 560001</p>
                </div>
                <div style='max-width:300px;'>
                    <h4>📧 Contact</h4>
                    <p>Email: <b>support@seedconnect.com</b><br>
                    Phone: +91 98765 43210<br>
                    Website: www.seedconnect.com</p>
                </div>
                <div style='max-width:300px;'>
                    <h4>🌐 Social</h4>
                    <p>Twitter: <b>@seedconnect</b><br>
                    LinkedIn: /company/seedconnect<br>
                    YouTube: SeedConnect Official</p>
                </div>
            </div>
            <br><p style='text-align:center;color:#6e8aa4;font-size:0.9rem;'>© 2025 SeedConnect Technologies Pvt. Ltd. — All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)


def startup_page(user):
    st.header("Startup Onboarding")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Company details")

    name = st.text_input("Company name", key="su_name")
    website = st.text_input("Website", key="su_website")
    contact_email = st.text_input("Contact email", key="su_email")
    target = st.number_input("Target funding (₹)", min_value=0.0, value=500000.0, step=10000.0, key="su_target")
    min_invest = st.number_input("Minimum investment unit (₹)", min_value=100.0, value=1000.0, step=100.0, key="su_mininv")

    st.markdown(
        "Please upload the following documents for company verification:\n"
        "- Certificate of Incorporation / Registration\n"
        "- Company PAN Card\n"
        "- Memorandum & Articles of Association (if applicable)\n"
        "- Board Resolution authorising this platform engagement\n"
        "- Proof of Registered Office Address (utility bill / lease / GST certificate)\n"
        "- Identity & Address Proof of the Director(s)/Partner(s)\n"
        "- Latest GST / Tax Registration certificate\n"
        "- Company logo (PNG or JPG format)"
    )

    docs = st.file_uploader(
        "Upload verification documents (you may attach multiple files)", 
        accept_multiple_files=True, type=["pdf","png","jpg","jpeg"]
    )
    logo = st.file_uploader("Company logo (PNG/JPG)", type=["png","jpg","jpeg"])
    financial_csv = st.file_uploader("Upload company financial data (.csv for ROI model)", type=["csv"])

    if st.button("Submit application"):
        if not name or not contact_email:
            st.warning("Provide company name and contact email.")
        elif not docs or len(docs) < 7:
            st.warning("Please upload at least 7 verification documents as listed.")
        elif logo is None:
            st.warning("Please upload a company logo.")
        else:
            pitch = {
                "id": len(st.session_state.pitches) + 1,
                "name": name,
                "website": website,
                "email": contact_email,
                "target": float(target),
                "min_invest": float(min_invest),
                "files": [],
                "logo": None,
                "financial_csv": None,
                "submitted_by": user["username"],
                "status": "Pending",
                "doc_verification": {f"name_{i}": "Pending" for i in range(1, 8)},
                "created_at": datetime.utcnow().isoformat()
            }

            for i, f in enumerate(docs, start=1):
                content = f.read()
                pitch["files"].append({
                    "name": f.name,
                    "content": content,
                    "type": f.type,
                    "idx": i
                })

            try:
                logo_bytes = logo.read()
                pitch["logo"] = {"name": logo.name, "content": logo_bytes, "type": logo.type}
            except Exception:
                pitch["logo"] = None

            if financial_csv is not None:
                try:
                    pitch["financial_csv"] = {
                        "name": financial_csv.name,
                        "content": financial_csv.read(),
                        "type": financial_csv.type
                    }
                    st.success("Financial data file attached successfully.")
                except Exception as e:
                    st.warning(f"Could not read CSV file: {e}")
            else:
                st.info("No financial data uploaded — investor ROI prediction will not be available.")

            st.session_state.pitches.append(pitch)
            st.success("Application submitted for compliance review.")

    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("My applications")
    mine = [p for p in st.session_state.pitches if p.get("submitted_by") == user["username"]]
    if not mine:
        st.info("No applications yet.")
    else:
        for p in mine:
            st.markdown(f'<div class="card"><h3>{html.escape(p["name"])} {status_badge_html(p.get("status"))}</h3>', unsafe_allow_html=True)
            st.write("Website:", p.get("website") or "-")
            st.write("Contact:", p.get("email"))
            st.write("Target:", f"₹{p.get('target', 0):,.2f}")
            st.write("Minimum investment:", f"₹{p.get('min_invest', 0):,.2f}")

            if p.get("logo"):
                try:
                    embed_image_bytes(p["logo"]["content"], width=160)
                except Exception:
                    st.write("Logo preview not available.")

            if p.get("files"):
                st.markdown("<b>Uploaded verification documents:</b>", unsafe_allow_html=True)
                for f in p["files"]:
                    st.write(f"Document {f['idx']}: {f['name']}")
                    if f['name'].lower().endswith(".pdf"):
                        embed_pdf_bytes(f['content'], height="320px")
                    else:
                        try:
                            embed_image_bytes(f['content'], width=240)
                        except Exception:
                            st.write("Preview not available.")

            if p.get("financial_csv"):
                st.markdown("**Financial data uploaded:** " + p["financial_csv"]["name"])

            st.markdown("</div>", unsafe_allow_html=True)

def checker_page(user):
    st.header("Compliance Review Dashboard")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    view = st.selectbox("View applications", ["Pending", "All"], key="checker_view")
    to_review = st.session_state.pitches if view == "All" else [p for p in st.session_state.pitches if p.get("status") != "Approved"]
    if not to_review:
        st.info("No applications to review.")
    else:
        for p in to_review:
            st.markdown(f'<div class="card"><h4>{html.escape(p["name"])} — {p.get("submitted_by")}</h4>', unsafe_allow_html=True)
            st.write("Email:", p.get("email"))
            st.write("Website:", p.get("website"))
            st.write("Target:", f"₹{p.get('target',0):,.2f}")
            st.write("Minimum investment:", f"₹{p.get('min_invest',0):,.2f}")
            st.write("Equity:", f"{p.get('equity',0):.2f}%")
            if p.get("logo"):
                st.markdown("<b>Logo</b>", unsafe_allow_html=True)
                try:
                    embed_image_bytes(p["logo"]["content"], width=180)
                except Exception:
                    st.write("Logo preview not available.")
            if p.get("files"):
                st.markdown("<b>Government documents</b>", unsafe_allow_html=True)
                cols = st.columns(2)
                for f in p["files"]:
                    with cols[(f["idx"]-1) % 2]:
                        st.write(f"Doc {f['idx']}: {f['name']}")
                        if f['name'].lower().endswith(".pdf"):
                            embed_pdf_bytes(f['content'], height="260px")
                        else:
                            try:
                                embed_image_bytes(f['content'], width=220)
                            except Exception:
                                st.write("Preview not available.")
                        status_key = f"doc_status_{p['id']}_{f['idx']}"
                        cur = p.get("doc_verification", {}).get(f"name_{f['idx']}","Pending")
                        choice = st.selectbox("Mark", ["Pending","Approved","Rejected"], index=["Pending","Approved","Rejected"].index(cur), key=status_key)
                        p["doc_verification"][f"name_{f['idx']}"] = choice
            verified = all(v == "Approved" for v in p.get("doc_verification", {}).values())
            cols2 = st.columns([1,1,1,1])
            if cols2[0].button(f"Approve Application-{p['id']}"):
                if verified:
                    p["status"] = "Approved"
                    st.success(f"Application {p['id']} approved.")
                else:
                    st.warning("All documents must be approved before approving the application.")
            if cols2[1].button(f"Reject Application-{p['id']}"):
                p["status"] = "Rejected"
                st.error(f"Application {p['id']} rejected.")
            if cols2[2].button(f"Request Info-{p['id']}"):
                p["status"] = "Pending"
                st.info(f"Requested more information for {p['id']}.")
            if cols2[3].button(f"Export Docs-{p['id']}"):
                files = []
                for f in p.get("files", []):
                    files.append({"name": f["name"], "size": len(f["content"]) if f.get("content") else 0})
                df = pd.DataFrame(files)
                csv = df_to_csv_bytes(df)
                st.download_button(f"Download manifest {p['id']}", csv, file_name=f"manifest_{p['id']}.csv")
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def investor_page(user):
    st.markdown("<h2 style='color:#06b6d4;'>💼 Investor Marketplace & Dashboard</h2>", unsafe_allow_html=True)
    st.markdown('<div class="card" style="padding:20px;">', unsafe_allow_html=True)

    # --- Search & Filters Section ---
    st.markdown("### 🔍 Find Investment Opportunities")
    st.markdown(
        "<p style='color:#9fb4c9;font-size:0.95rem;'>Filter verified startups by approval status, target range, or search by name.</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1])
    with col1:
        q = st.text_input("Search by startup name or website", placeholder="e.g. SeedConnect Tech", key="inv_search")
    with col2:
        status_filter = st.selectbox("Approval Status", ["Any", "Approved", "Pending", "Rejected"], key="inv_status")
    with col3:
        sort_by = st.selectbox("Sort by", ["Relevance", "Target: Low→High", "Target: High→Low"], key="inv_sort")
    with col4:
        page_size = st.selectbox("Results per page", [6, 12, 24], index=0, key="inv_pagesize")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    results = st.session_state.pitches

    # --- Filter logic ---
    if q:
        results = [
            p for p in results
            if q.lower() in (p.get("name") or "").lower()
            or q.lower() in (p.get("website") or "").lower()
        ]
    if status_filter != "Any":
        results = [p for p in results if p.get("status", "").lower() == status_filter.lower()]
    if sort_by == "Target: Low→High":
        results = sorted(results, key=lambda x: x.get("target", 0))
    elif sort_by == "Target: High→Low":
        results = sorted(results, key=lambda x: x.get("target", 0), reverse=True)

    # --- Results Display ---
    if not results:
        st.info("No startups match your search filters.")
    else:
        st.markdown('<div class="grid">', unsafe_allow_html=True)
        for p in results[:page_size]:
            st.markdown(f'<div class="product-card"><h4>{html.escape(p["name"])}</h4>', unsafe_allow_html=True)
            st.markdown(f'<div class="small-muted">Submitted by {html.escape(p.get("submitted_by", "-"))}</div>', unsafe_allow_html=True)
            st.write("Target funding:", f"₹{p.get('target', 0):,.2f}")
            st.write("Minimum investment:", f"₹{p.get('min_invest', 0):,.2f}")
            st.markdown(status_badge_html(p.get("status")))

            if p.get("logo"):
                try:
                    embed_image_bytes(p["logo"]["content"], width=120)
                except Exception:
                    st.write("Logo not available.")

            col_a, col_b = st.columns([1, 1])
            with col_a:
                amount = st.number_input(
                    f"Amount to invest-{p['id']}",
                    min_value=float(p.get("min_invest", 100.0)),
                    value=float(p.get("min_invest", 1000.0)),
                    step=float(p.get("min_invest", 100.0)),
                    key=f"amt_{p['id']}"
                )
            with col_b:
                if st.button(f"Add to Cart-{p['id']}"):
                    if p.get("status", "").lower() != "approved":
                        st.warning("Only approved startups can be invested in.")
                    else:
                        st.session_state.cart.append({
                            "pitch_id": p["id"],
                            "name": p["name"],
                            "amount": float(amount),
                            "added_at": datetime.utcnow().isoformat()
                        })
                        st.success(f"Added ₹{float(amount):,.2f} to cart for {p['name']}")

            # --- Details + ROI Forecast ---
            if st.button(f"View Details-{p['id']}"):
                with st.expander(f"Details — {p['name']}"):
                    st.write("Website:", p.get("website") or "-")
                    st.write("Contact:", p.get("email"))
                    st.write("Target:", f"₹{p.get('target', 0):,.2f}")
                    st.write("Minimum investment:", f"₹{p.get('min_invest', 0):,.2f}")

                    st.markdown("### 📈 ROI Prediction (6-Month ARIMA Forecast)")
                    if p.get("financial_csv"):
                        try:
                            df = safe_read_csv(BytesIO(p["financial_csv"]["content"]))
                            dates, forecast, conf, roi = predict_forecast_and_roi(df)
                            st.success(f"Predicted ROI over next 6 months: **{roi:.2f}%**")

                            if dates is not None:
                                fig, ax = plt.subplots()
                                ax.plot(df["date"], df["value"], label="Historical", linewidth=2)
                                ax.plot(dates, forecast, linestyle="--", marker="o", label="Forecast")
                                if conf is not None:
                                    ax.fill_between(dates, conf.iloc[:, 0], conf.iloc[:, 1], alpha=0.2)
                                ax.set_title(f"{p['name']} ROI Forecast")
                                ax.set_xlabel("Date")
                                ax.set_ylabel("Value")
                                ax.legend()
                                st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Could not process uploaded CSV: {e}")
                    else:
                        st.info("No financial data CSV uploaded for this startup.")

                    if p.get("files"):
                        st.markdown("### 📂 Verification Documents")
                        for f in p["files"]:
                            st.write(f["name"])
                            if f["name"].lower().endswith(".pdf"):
                                embed_pdf_bytes(f["content"], height="240px")
                            elif any(f["name"].lower().endswith(x) for x in [".png", ".jpg", ".jpeg"]):
                                try:
                                    embed_image_bytes(f["content"], width=220)
                                except Exception:
                                    st.write("Preview not available.")

            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Wallet & Cart Section ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    col_wallet, col_cart = st.columns([1, 2])
    with col_wallet:
        st.markdown('<div class="card" style="padding:15px;">', unsafe_allow_html=True)
        st.subheader("💰 Wallet Balance")
        wallet = st.session_state.users.get(user["username"], {}).get("wallet", 0.0)
        st.metric("Current Balance", f"₹{wallet:,.2f}")

        add_amount = st.number_input("Add funds to wallet (₹)", min_value=100.0, value=1000.0, step=100.0, key="add_funds")
        if st.button("Add Money"):
            st.session_state.users[user["username"]]["wallet"] = wallet + add_amount
            st.success(f"₹{add_amount:,.2f} added successfully!")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_cart:
        st.markdown('<div class="card" style="padding:15px;">', unsafe_allow_html=True)
        st.subheader("🛒 Investment Cart")
        if not st.session_state.cart:
            st.info("Your cart is currently empty.")
        else:
            total = sum(item["amount"] for item in st.session_state.cart)
            for i, item in enumerate(st.session_state.cart, start=1):
                st.markdown(f"**{i}. {item['name']}** — ₹{item['amount']:,.2f}")
                if st.button(f"Remove-{i}", key=f"remove_{i}"):
                    st.session_state.cart.pop(i - 1)
                    st.experimental_rerun()
            st.markdown(f"<br><b>Total Investment:</b> ₹{total:,.2f}", unsafe_allow_html=True)
            if st.button("💳 Proceed to Checkout"):
                if wallet >= total:
                    st.session_state.users[user["username"]]["wallet"] = wallet - total
                    for item in st.session_state.cart:
                        st.session_state.investments.append({
                            "investor": user["username"],
                            "pitch_id": item["pitch_id"],
                            "amount": item["amount"],
                            "date": datetime.utcnow().isoformat()
                        })
                    order = {
                        "investor": user["username"],
                        "items": list(st.session_state.cart),
                        "total": total,
                        "date": datetime.utcnow().isoformat()
                    }
                    st.session_state.orders.append(order)
                    st.session_state.cart = []
                    st.success(f"Checkout successful. You invested ₹{total:,.2f}.")
                else:
                    st.error("Insufficient balance. Please add funds to wallet.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Investor Dashboard ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📊 My Investment Dashboard")

    mine = [inv for inv in st.session_state.investments if inv["investor"] == user["username"]]
    if not mine:
        st.info("No investments made yet.")
    else:
        df = pd.DataFrame(mine)
        df["date"] = pd.to_datetime(df["date"])
        total_invest = df["amount"].sum()
        st.metric("Total Invested", f"₹{total_invest:,.2f}")
        st.markdown("#### Recent Investments")
        for inv in sorted(mine, key=lambda x: x["date"], reverse=True):
            p = next((x for x in st.session_state.pitches if x["id"] == inv["pitch_id"]), None)
            st.markdown(
                f'<div class="card"><b>{p["name"] if p else "Unknown"}</b><br>'
                f'Invested ₹{inv["amount"]:,.2f} on {inv["date"]}</div>',
                unsafe_allow_html=True
            )

        # Optional visualization
        fig, ax = plt.subplots()
        ax.plot(df["date"], df["amount"].cumsum(), marker="o", linewidth=2)
        ax.set_title("Cumulative Investment Growth")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Invested (₹)")
        st.pyplot(fig)


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

if st.session_state.page == "home" or st.session_state.current_user is None:
    landing_page()
else:
    main_app()



