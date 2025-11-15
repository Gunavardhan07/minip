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
import base64
import os
import json

DB_DIR = "database"
PITCHES_FILE = os.path.join(DB_DIR, "pitches.json")
INVESTMENTS_FILE = os.path.join(DB_DIR, "investments.json")


def ensure_db():
    """Create database folder and file if missing."""
    try:
        if not os.path.exists(DB_DIR):
            os.makedirs(DB_DIR, exist_ok=True)
        if not os.path.exists(PITCHES_FILE):
            with open(PITCHES_FILE, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)
    except Exception as e:
        # don't crash app; just log to console
        print("ensure_db error:", e)

def load_pitches_from_file():
    """Return list of pitch dicts from JSON file (empty list if missing/invalid)."""
    ensure_db()
    try:
        with open(PITCHES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception as e:
        print("load_pitches_from_file error:", e)
        return []

def save_pitches_to_file(pitches_list):
    """Overwrite JSON file with current pitches_list (list of dicts)."""
    ensure_db()
    try:
        with open(PITCHES_FILE, "w", encoding="utf-8") as f:
            json.dump(pitches_list, f, indent=2, default=str)
        return True
    except Exception as e:
        print("save_pitches_to_file error:", e)
        return False
def load_investments_from_file():
    ensure_db()
    if not os.path.exists(INVESTMENTS_FILE):
        with open(INVESTMENTS_FILE, "w") as f:
            json.dump([], f)
        return []
    try:
        with open(INVESTMENTS_FILE, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except:
        return []

def save_investments_to_file(inv_list):
    ensure_db()
    try:
        with open(INVESTMENTS_FILE, "w") as f:
            json.dump(inv_list, f, indent=2, default=str)
        return True
    except:
        return False



def get_image_as_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


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
    """
    Reads a CSV with multiple financial attributes.
    Required: A 'date' column + at least 2 numeric columns.
    """
    df = pd.read_csv(file)

    # Force date column
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Select numeric columns except date
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 1:
        raise ValueError("CSV must contain at least one numeric financial metric.")

    return df


def predict_forecast_and_roi(df: pd.DataFrame, steps: int = 6):
    """
    Multivariate ARIMAX forecasting using all numeric features in CSV.
    Uses 'profit' as primary target if available, else first numeric column.
    """
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Target variable
        if "profit" in num_cols:
            target_col = "profit"
        else:
            target_col = num_cols[0]

        y = df[target_col]

        # Exogenous features (other numeric variables)
        exog_cols = [c for c in num_cols if c != target_col]
        exog = df[exog_cols] if exog_cols else None

        # Fit ARIMAX model
        model = sm.tsa.ARIMA(y, exog=exog, order=(2,1,2))
        res = model.fit()

        # Create future exogenous (simple forward-fill)
        if exog is not None:
            last_exog = exog.iloc[-1:]
            future_exog = pd.concat([last_exog] * steps, ignore_index=True)
        else:
            future_exog = None

        # Forecast
        forecast = res.forecast(steps=steps, exog=future_exog)
        conf = res.get_forecast(steps=steps, exog=future_exog).conf_int()

        # Future dates
        future_dates = pd.date_range(df["date"].iloc[-1], periods=steps + 1, freq="M")[1:]

        # ROI
        latest = y.iloc[-1]
        mean_forecast = np.mean(forecast)
        roi = (mean_forecast / latest - 1) * 100.0

        return future_dates, np.array(forecast), conf, roi

    except Exception as e:
        print("ARIMAX failed:", e)
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
    
ensure_db()
if "pitches" not in st.session_state:
    st.session_state.pitches = load_pitches_from_file()
if "investments" not in st.session_state:
    st.session_state.investments = load_investments_from_file()

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
   
    bg_base64 = get_image_as_base64("images/landing_header.png")

    st.markdown(
        f"""
    <div style="background-image: url('data:image/jpg;base64,{bg_base64}');
                background-size: cover;
                background-position: center;
                padding: 220px 20px;
                border-radius: 20px;
                text-align: center;">
    

    </div>
    """,
        unsafe_allow_html=True
    )


   
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
    # --- HELP / INSTRUCTIONS BOX ---
    with st.expander("📘 HELP — How to Apply as a Startup (Read Before Submitting)"):
        st.markdown("""
        ### 🌱 **Startup Application Guide**
        Follow these steps carefully to ensure your application is processed smoothly:

        ---
        ### **1️⃣ Enter Company Information**
        Provide:
        - Company Name  
        - Official Website (optional)  
        - Official Contact Email  
        - Target Funding Amount  
        - Minimum Investment Amount  

        Make sure your contact details are accurate.

        ---
        ### **2️⃣ Upload All Required KYC Verification Documents**
        Upload **minimum 7 documents**, including:
        - Certificate of Incorporation / Registration  
        - Company PAN Card  
        - MOA / AOA  
        - Board Resolution  
        - Registered Office Address Proof  
        - Director(s) ID & Address Proof  
        - Latest GST / Tax Certificate  
        
        Documents can be **PDF / JPG / PNG**.

        ---
        ### **3️⃣ Upload Your Company Logo**
        This will be shown to investors on the marketplace.

        ---
        ### **4️⃣ Upload Financial CSV (Optional but Strongly Recommended)**
        Upload a CSV containing:
        - date  
        - revenue  
        - expenses  
        - profit  
        - cashflow  
        - users  
        - churn  
        
        This data is used for **ROI forecasting** using the ARIMAX model.

        ---
        ### **5️⃣ Submit Your Application**
        Click **Submit Application for Review**.

        ---
        ### **6️⃣ Track Your Application**
        Scroll down to **My Applications**.
        """)

    # --- HEADER ---
    st.markdown("""
        <div style='background:linear-gradient(90deg,#071A2A,#09344E);padding:40px 20px;border-radius:12px;
                    box-shadow:0 8px 24px rgba(0,0,0,0.45);text-align:center;'>
            <h1 style='color:#06b6d4;font-size:2.5rem;margin-bottom:8px;'>🏢 Startup Onboarding</h1>
            <p style='color:#9fb4c9;font-size:1.05rem;margin:0;'>Submit your verified documents & financials.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- COMPANY DETAILS ---
    st.markdown("""
        <div style='background:rgba(255,255,255,0.03);padding:25px;border-radius:10px;
                    box-shadow:0 4px 16px rgba(0,0,0,0.4);'>
            <h2 style='color:#06b6d4;'>🏗️ Company Information</h2>
            <p style='color:#9fb4c9;'>Provide basic details about your company.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    name = st.text_input("Company Name", key="su_name")
    website = st.text_input("Website (if available)", key="su_website")
    contact_email = st.text_input("Official Contact Email", key="su_email")

    col1, col2 = st.columns(2)
    with col1:
        target = st.number_input("Target Funding (₹)", min_value=0.0, value=500000.0, step=10000.0)
    with col2:
        min_invest = st.number_input("Minimum Investment Unit (₹)", min_value=100.0, value=1000.0, step=100.0)

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- DOCUMENT UPLOAD ---
    st.markdown("""
        <div style='background:rgba(255,255,255,0.02);padding:25px;border-radius:10px;
                    box-shadow:0 4px 16px rgba(0,0,0,0.4);'>
            <h2 style='color:#06b6d4;'>📄 KYC & Compliance Documents</h2>
            <p style='color:#9fb4c9;margin-bottom:15px;'>Upload minimum 7 documents.</p>
        </div>
    """, unsafe_allow_html=True)

    docs = st.file_uploader("Upload Verification Documents", accept_multiple_files=True,
                            type=["pdf", "png", "jpg", "jpeg"])
    logo = st.file_uploader("Upload Company Logo", type=["png", "jpg", "jpeg"])
    financial_csv = st.file_uploader("Upload Financial CSV (optional)", type=["csv"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------
    #       SUBMIT APPLICATION
    # ---------------------------
    if st.button("🚀 Submit Application for Review"):

        # --- VALIDATIONS ---
        if not name or not contact_email:
            st.warning("Please provide company name and contact email.")
            return

        if not docs or len(docs) < 7:
            st.warning("Upload at least 7 verification documents.")
            return

        if logo is None:
            st.warning("Please upload a company logo.")
            return

        # --------------------------------------
        # Build pitch object
        # --------------------------------------
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

        # Store document files
        for i, f in enumerate(docs, start=1):
            pitch["files"].append({
                "name": f.name,
                "content": f.read(),
                "type": f.type,
                "idx": i
            })

        # Logo
        try:
            pitch["logo"] = {
                "name": logo.name,
                "content": logo.read(),
                "type": logo.type
            }
        except:
            pitch["logo"] = None

        # Financial CSV
        if financial_csv:
            try:
                pitch["financial_csv"] = {
                    "name": financial_csv.name,
                    "content": financial_csv.read(),
                    "type": financial_csv.type
                }
                st.success("✅ Financial CSV uploaded.")
            except Exception as e:
                st.warning(f"Error reading CSV: {e}")
        else:
            st.info("No financial CSV uploaded.")

        # --------------------------------------
        # Save pitch (JSON DATABASE)
        # --------------------------------------
        pitches = st.session_state.pitches.copy()
        pitches.append(pitch)

        if save_pitches_to_file(pitches):
            st.session_state.pitches = pitches
            st.success("🎉 Application submitted successfully!")
        else:
            st.error("❌ Failed to save application. Try again.")

        st.markdown("<br><hr>", unsafe_allow_html=True)

    # ---------------------------
    #     LIST MY APPLICATIONS
    # ---------------------------
    st.markdown("""
        <div style='background:rgba(255,255,255,0.03);padding:25px;border-radius:10px;
                    box-shadow:0 4px 16px rgba(0,0,0,0.4);'>
            <h2 style='color:#06b6d4;'>📂 My Applications</h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    mine = [p for p in st.session_state.pitches if p.get("submitted_by") == user["username"]]

    if not mine:
        st.info("You have not submitted any startup applications yet.")
        return

    # Application cards
    for p in mine:
        st.markdown(f"""
            <div style='padding:18px;margin-bottom:16px;border-radius:10px;
                        box-shadow:0 4px 16px rgba(0,0,0,0.45);'>
                <h3 style='color:#06b6d4;'>{html.escape(p["name"])} {status_badge_html(p["status"])}</h3>
                <p style='color:#9fb4c9;'>Submitted by: <b>{p["submitted_by"]}</b></p>
                <p style='color:#cbd6e0;'>Website: {p["website"] or "-"}</p>
                <p style='color:#cbd6e0;'>Contact: {p["email"]}</p>
                <p style='color:#cbd6e0;'>Target: ₹{p["target"]:,.2f}</p>
                <p style='color:#cbd6e0;'>Minimum Investment: ₹{p["min_invest"]:,.2f}</p>
        """, unsafe_allow_html=True)

        # Logo
        if p.get("logo"):
            st.markdown("<b>Logo:</b>", unsafe_allow_html=True)
            try:
                embed_image_bytes(p["logo"]["content"], width=160)
            except:
                st.write("Logo preview unavailable.")

        # Documents
        if p.get("files"):
            st.markdown("<b>Uploaded Verification Documents:</b>", unsafe_allow_html=True)
            cols = st.columns(2)
            for f in p["files"]:
                with cols[(f["idx"] - 1) % 2]:
                    st.write(f"📎 {f['name']}")
                    if f["name"].lower().endswith(".pdf"):
                        embed_pdf_bytes(f["content"], height="320px")
                    else:
                        try:
                            embed_image_bytes(f["content"], width=240)
                        except:
                            st.write("Preview unavailable.")

        # Financial CSV
        if p.get("financial_csv"):
            st.markdown(f"**Financial Data:** {p['financial_csv']['name']}", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)



def checker_page(user):
    st.header("Compliance Review Dashboard")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    view = st.selectbox("View applications", ["Pending", "All"], key="checker_view")

    # Filter applications
    if view == "All":
        to_review = st.session_state.pitches
    else:
        to_review = [p for p in st.session_state.pitches if p.get("status") != "Approved"]

    if not to_review:
        st.info("No applications to review.")
        return

    for p in to_review:
        st.markdown(f'<div class="card"><h4>{html.escape(p["name"])} — {p["submitted_by"]}</h4>', unsafe_allow_html=True)
        st.write("Email:", p.get("email"))
        st.write("Website:", p.get("website"))
        st.write("Target:", f"₹{p.get('target', 0):,.2f}")
        st.write("Minimum investment:", f"₹{p.get('min_invest', 0):,.2f}")

        # Logo
        if p.get("logo"):
            st.markdown("<b>Logo</b>", unsafe_allow_html=True)
            try:
                img = Image.open(BytesIO(p["logo"]["content"]))
                img.verify()
                img = Image.open(BytesIO(p["logo"]["content"]))
                st.image(img, width=180)
            except:
                st.write("Logo preview not available.")

        # Documents
        if p.get("files"):
            st.markdown("<b>Government documents</b>", unsafe_allow_html=True)
            cols = st.columns(2)

            for f in p["files"]:
                with cols[(f["idx"] - 1) % 2]:
                    st.write(f"Doc {f['idx']}: {f['name']}")

                    fname = f["name"].lower()

                    # PDF preview
                    if fname.endswith(".pdf"):
                        try:
                            embed_pdf_bytes(f["content"], height="260px")
                        except:
                            st.write("PDF preview error.")
                    else:
                        # Safe image preview
                        try:
                            img = Image.open(BytesIO(f["content"]))
                            img.verify()
                            img = Image.open(BytesIO(f["content"]))
                            st.image(img, width=220)
                        except:
                            st.write("Preview not available.")

                    # Verification status dropdown
                    status_key = f"doc_status_{p['id']}_{f['idx']}"
                    current_status = p.get("doc_verification", {}).get(f"name_{f['idx']}", "Pending")

                    p["doc_verification"][f"name_{f['idx']}"] = st.selectbox(
                        "Mark",
                        ["Pending", "Approved", "Rejected"],
                        index=["Pending", "Approved", "Rejected"].index(current_status),
                        key=status_key
                    )

        # Determine if all docs approved
        verified = all(v == "Approved" for v in p["doc_verification"].values())

        cols2 = st.columns([1, 1, 1, 1])

        # --- APPROVE ---
        if cols2[0].button(f"Approve Application-{p['id']}"):
            if verified:
                p["status"] = "Approved"
                save_pitches_to_file(st.session_state.pitches)
                st.success(f"Application {p['id']} approved.")
                st.rerun()
            else:
                st.warning("All documents must be approved first.")

        # --- REJECT ---
        if cols2[1].button(f"Reject Application-{p['id']}"):
            p["status"] = "Rejected"
            save_pitches_to_file(st.session_state.pitches)
            st.error(f"Application {p['id']} rejected.")
            st.rerun()

        # --- REQUEST MORE INFO ---
        if cols2[2].button(f"Request Info-{p['id']}"):
            p["status"] = "Pending"
            save_pitches_to_file(st.session_state.pitches)
            st.info(f"Requested more information for {p['id']}.")
            st.rerun()

        # --- EXPORT DOC LIST ---
        if cols2[3].button(f"Export Docs-{p['id']}"):
            files = [
                {"name": f["name"], "size": len(f["content"])}
                for f in p.get("files", [])
            ]
            df = pd.DataFrame(files)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"Download manifest {p['id']}",
                csv,
                file_name=f"manifest_{p['id']}.csv"
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def investor_page(user):

    # ---------------------------
    # HELP SECTION
    # ---------------------------
    with st.expander("📘 HELP — How to Use the Investor Dashboard"):
        st.markdown("""
        ### 💼 Investor Guide
        - Browse approved startups  
        - View ROI forecasts  
        - Add investments to your cart  
        - Pay using wallet balance  
        - Track your portfolio with a growth chart  
        """)

    # ---------------------------
    # HEADER WITH BACKGROUND IMAGE
    # ---------------------------
    try:
        bg_base64 = get_image_as_base64("images/investor_bg.png")
    except Exception:
        bg_base64 = ""

    st.markdown(f"""
        <div style="background-image: url('data:image/jpg;base64,{bg_base64}');
                    background-size: cover; background-position:center;
                    padding: 70px 20px; border-radius:18px;
                    text-align:center; box-shadow:0 8px 24px rgba(0,0,0,0.55);">
            <h1 style="color:#00d0ff;font-size:3rem;font-weight:700;
                       text-shadow:2px 2px 8px rgba(0,0,0,0.7);">
                💼 Investor Marketplace
            </h1>
            <p style="color:#e3edf5;font-size:1.2rem;">
                Discover verified startups, forecast ROI, and grow your portfolio.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------
    # SEARCH & FILTERS
    # ---------------------------
    col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1])
    with col1:
        q = st.text_input("Search Startup", placeholder="Name or website")
    with col2:
        status_filter = st.selectbox("Status", ["Any", "Approved", "Pending", "Rejected"])
    with col3:
        sort_by = st.selectbox("Sort", ["Relevance", "Target: Low→High", "Target: High→Low"])
    with col4:
        page_size = st.selectbox("Results per page", [6, 12, 24])

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------------------
    # FILTER STARTUPS
    # ---------------------------
    results = st.session_state.pitches

    if q:
        results = [p for p in results if q.lower() in p.get("name", "").lower()
                                     or q.lower() in p.get("website", "").lower()]

    if status_filter != "Any":
        results = [p for p in results if p.get("status", "").lower() == status_filter.lower()]

    if sort_by == "Target: Low→High":
        results = sorted(results, key=lambda x: x.get("target", 0))
    elif sort_by == "Target: High→Low":
        results = sorted(results, key=lambda x: x.get("target", 0), reverse=True)

    # ---------------------------
    # DISPLAY GRID OF STARTUPS
    # ---------------------------
    if not results:
        st.info("No startups match your filters.")
        return

    st.markdown("<div class='grid'>", unsafe_allow_html=True)

    for p in results[:page_size]:

        invested_amount = sum(inv["amount"] for inv in st.session_state.investments
                              if inv["pitch_id"] == p["id"])
        progress = min(invested_amount / p.get("target", 1), 1.0)

        st.markdown(f"""
            <div class='product-card' style='padding:18px;border-radius:12px;
                    box-shadow:0 6px 18px rgba(0,0,0,0.45);'>
                <h3 style='color:#06b6d4;'>{html.escape(p['name'])}</h3>
                <p class='small-muted'>Submitted by {p.get('submitted_by')}</p>
                <p style='color:#cbd6e0;'>Target: ₹{p['target']:,.2f}</p>
                <p style='color:#cbd6e0;'>Min Invest: ₹{p['min_invest']:,.2f}</p>
                {status_badge_html(p['status'])}
        """, unsafe_allow_html=True)

        # Progress Bar
        st.progress(progress)
        st.markdown(
            f"<p style='color:#9fb4c9;'>Raised ₹{invested_amount:,.2f} / ₹{p['target']:,.2f}</p>",
            unsafe_allow_html=True
        )

        # Logo
        if p.get("logo"):
            try:
                embed_image_bytes(p["logo"]["content"], width=120)
            except Exception:
                st.write("Logo unavailable")

        # INVEST SECTION
        colA, colB = st.columns([1, 1])
        with colA:
            amount = st.number_input(
                f"Amount-{p['id']}",
                min_value=float(p["min_invest"]),
                value=float(p["min_invest"]),
                step=float(p["min_invest"]),
                key=f"amt_{p['id']}"
            )

        with colB:
            if st.button(f"Add to Cart-{p['id']}"):
                if p["status"].lower() != "approved":
                    st.warning("Only approved startups can be invested in.")
                else:
                    st.session_state.cart.append({
                        "pitch_id": p["id"],
                        "name": p["name"],
                        "amount": float(amount),
                        "added_at": datetime.utcnow().isoformat()
                    })
                    st.success("Added to cart!")

        # DETAILS (with corrected ARIMA plotting)
        if st.button(f"Details-{p['id']}"):
            with st.expander(f"📘 {p['name']} — Details"):
                st.write("**Website:**", p.get("website"))
                st.write("**Contact:**", p.get("email"))
                st.write("**Target Funding:**", f"₹{p['target']:,.2f}")

                # --- ROI forecasting: handle CSV and plotting robustly ---
                st.markdown("### 📈 ROI Forecast")
                if p.get("financial_csv"):
                    try:
                        # safe_read_csv expects a file-like; ensure bytes
                        content = p["financial_csv"]["content"]
                        # If content is str (old bad data), convert to bytes
                        if isinstance(content, str):
                            content = content.encode("utf-8")

                        df = safe_read_csv(BytesIO(content))  # returns dataframe with date + numeric cols

                        # Determine target column for plotting & forecasting
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if not numeric_cols:
                            st.warning("Financial CSV contains no numeric columns for forecasting.")
                        else:
                            # Choose 'profit' if available, otherwise first numeric column
                            if "profit" in numeric_cols:
                                plot_col = "profit"
                            else:
                                plot_col = numeric_cols[0]

                            # Run forecast using your existing helper
                            dates, forecast, conf, roi = predict_forecast_and_roi(df, steps=6)

                            # Show ROI metric (even if forecast failed, roi may be 0.0)
                            st.success(f"Predicted 6-Month ROI: **{roi:.2f}%**")

                            # Plot historical and forecast if available
                            fig, ax = plt.subplots()

                            # Historical plot of the chosen target
                            ax.plot(df["date"], df[plot_col], label="Historical", linewidth=2)

                            if dates is not None and forecast is not None:
                                # forecast is an array aligned with 'dates'
                                ax.plot(dates, forecast, linestyle="--", marker="o", label="Forecast")

                                # draw confidence interval if conf is provided and has correct shape
                                try:
                                    if conf is not None and hasattr(conf, "iloc"):
                                        lower = conf.iloc[:, 0].values
                                        upper = conf.iloc[:, 1].values
                                        ax.fill_between(dates, lower, upper, alpha=0.2)
                                except Exception:
                                    pass

                            ax.set_title(f"{p['name']} — {plot_col} Forecast")
                            ax.set_xlabel("Date")
                            ax.set_ylabel(plot_col.capitalize())
                            ax.legend()
                            st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not process financial CSV: {e}")
                else:
                    st.info("No financial data uploaded for this startup.")

                # -----------------------------
                # SAFE DOCUMENT PREVIEW
                # -----------------------------
                st.markdown("### 📂 Verification Documents")
                for f in p.get("files", []):
                    st.write(f["name"])
                    # handle string vs bytes
                    data = f.get("content")
                    if isinstance(data, str):
                        try:
                            data = data.encode("latin1")
                        except Exception:
                            st.info("Invalid file content — cannot preview.")
                            continue

                    if f["name"].lower().endswith(".pdf"):
                        try:
                            embed_pdf_bytes(data, height="240px")
                        except Exception:
                            st.info("PDF could not be displayed.")
                    else:
                        # try PIL open for images
                        try:
                            img = Image.open(BytesIO(data))
                            img.verify()
                            img = Image.open(BytesIO(data))
                            st.image(img, width=220)
                        except Exception:
                            st.info("Preview unavailable for this file type.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------------------
    # WALLET AND CART
    # ---------------------------
    colW, colC = st.columns([1, 2])

    # WALLET
    with colW:
        st.subheader("💼 Wallet")
        wallet = st.session_state.users[user["username"]]["wallet"]
        st.metric("Balance", f"₹{wallet:,.2f}")

        add_money = st.number_input("Add Funds", min_value=100.0, value=1000.0)
        if st.button("➕ Add"):
            st.session_state.users[user["username"]]["wallet"] += add_money
            st.success("Funds added successfully")

    # CART
    with colC:
        st.subheader("🛒 Investment Cart")
        if not st.session_state.cart:
            st.info("Cart is empty.")
        else:
            total = sum(x["amount"] for x in st.session_state.cart)
            for i, item in enumerate(st.session_state.cart, 1):
                st.write(f"**{i}. {item['name']} — ₹{item['amount']:,.2f}**")
                if st.button(f"Remove-{i}"):
                    st.session_state.cart.pop(i-1)
                    st.rerun()

            st.write(f"**Total: ₹{total:,.2f}**")

            # CHECKOUT
            if st.button("💳 Checkout"):
                if wallet < total:
                    st.error("Insufficient wallet balance.")
                else:
                    st.session_state.users[user["username"]]["wallet"] -= total

                    inv_list = st.session_state.investments.copy()
                    for item in st.session_state.cart:
                        inv_list.append({
                            "id": len(inv_list) + 1,
                            "investor": user["username"],
                            "pitch_id": item["pitch_id"],
                            "startup_name": item["name"],
                            "amount": float(item["amount"]),
                            "date": datetime.utcnow().isoformat()
                        })

                    if save_investments_to_file(inv_list):
                        st.session_state.investments = inv_list
                        st.session_state.cart.clear()
                        st.success("Investment successful & saved to database!")
                    else:
                        st.error("Failed to save investment.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------------------
    # INVESTOR DASHBOARD — GRAPH
    # ---------------------------
    st.subheader("📊 My Investment Dashboard")

    mine = [x for x in st.session_state.investments if x["investor"] == user["username"]]

    if not mine:
        st.info("You have no investments yet.")
        return

    df_inv = pd.DataFrame(mine)
    df_inv["date"] = pd.to_datetime(df_inv["date"])

    st.metric("Total Invested", f"₹{df_inv['amount'].sum():,.2f}")

    fig, ax = plt.subplots()
    ax.plot(df_inv["date"], df_inv["amount"].cumsum(), marker="o", linewidth=2)
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












































