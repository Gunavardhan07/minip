import streamlit as st
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
from io import StringIO

# --- Page Config ---
st.set_page_config(page_title="CrowdPitch Pro", page_icon="🚀", layout="wide")

# --- Custom CSS for styling ---
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1E1E2E, #232946);
    color: white;
}
section.main > div {
    background-color: #1E1E2E;
    color: white;
    padding: 1.5rem;
    border-radius: 1rem;
}
.sidebar .sidebar-content {
    background: linear-gradient(135deg, #141820, #212A3E);
    color: white;
}
.stButton>button {
    background-color: #00ADB5;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1rem;
    font-size: 1rem;
}
.stButton>button:hover {
    background-color: #06d6a0;
    color: black;
}
</style>
""", unsafe_allow_html=True)

# --- Initialize session state ---
if "pitches" not in st.session_state:
    st.session_state.pitches = []

# --- App Header ---
st.title("🚀 CrowdPitch Pro")
st.markdown("### Empowering Startups. Inspiring Investors.")
st.write("A smart pitch prediction platform using ARIMA forecasting and interactive visuals.")

# --- Tabs for Startup / Investor ---
tab1, tab2 = st.tabs(["🏢 Startup Zone", "💼 Investor Zone"])

# ============================================================
# 🏢 STARTUP ZONE
# ============================================================
with tab1:
    st.header("📈 Create Your Startup Pitch")

    startup_name = st.text_input("Startup Name")
    description = st.text_area("Startup Description")
    video_url = st.text_input("YouTube Pitch Video URL")
    uploaded_file = st.file_uploader("Upload your performance CSV (date, value)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        st.subheader("📊 Historical Performance")
        st.line_chart(df.set_index("date")["value"])

        # --- ARIMA Forecasting ---
        st.info("Running ARIMA model for forecasting future performance...")
        model = pm.auto_arima(df["value"], seasonal=False, suppress_warnings=True)
        n_periods = 6
        forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
        future_dates = pd.date_range(df["date"].iloc[-1], periods=n_periods + 1, freq="M")[1:]

        forecast_df = pd.DataFrame({
            "date": future_dates,
            "forecast": forecast,
            "lower": conf_int[:, 0],
            "upper": conf_int[:, 1]
        })

        st.subheader("🔮 Predicted Future Trend (Next 6 Months)")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["date"], df["value"], label="Historical", marker="o", color="#00ADB5")
        ax.plot(forecast_df["date"], forecast_df["forecast"], label="Forecast", marker="x", color="#FFD369")
        ax.fill_between(forecast_df["date"], forecast_df["lower"], forecast_df["upper"], color="#FFD369", alpha=0.3)
        ax.legend()
        ax.set_facecolor("#222831")
        st.pyplot(fig)

        # --- Save Pitch to Session ---
        if st.button("💾 Save My Pitch"):
            pitch = {
                "name": startup_name,
                "desc": description,
                "video": video_url,
                "data": df,
                "forecast": forecast_df
            }
            st.session_state.pitches.append(pitch)
            st.success(f"✅ Pitch '{startup_name}' saved successfully!")

# ============================================================
# 💼 INVESTOR ZONE
# ============================================================
with tab2:
    st.header("💼 Explore Startup Pitches")

    if len(st.session_state.pitches) == 0:
        st.warning("No pitches yet! Ask startups to upload their data first.")
    else:
        pitch_names = [p["name"] for p in st.session_state.pitches]
        choice = st.selectbox("Choose a Startup to View:", pitch_names)

        selected = next(p for p in st.session_state.pitches if p["name"] == choice)
        st.subheader(f"🚀 {selected['name']}")
        st.write(selected["desc"])

        # --- Show Forecast Chart ---
        df = selected["data"]
        fdf = selected["forecast"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["date"], df["value"], label="Historical", marker="o", color="#00ADB5")
        ax.plot(fdf["date"], fdf["forecast"], label="Forecast", marker="x", color="#FFD369")
        ax.fill_between(fdf["date"], fdf["lower"], fdf["upper"], color="#FFD369", alpha=0.3)
        ax.legend()
        ax.set_facecolor("#222831")
        st.pyplot(fig)

        # --- Video Pitch ---
        if selected["video"]:
            st.subheader("🎥 Watch the Pitch Video")
            st.video(selected["video"])

        # --- Key Metrics ---
        st.subheader("📈 Forecast Summary")
        st.write(fdf)

st.markdown("---")
st.markdown("<center>💡 Built with ❤️ using Streamlit | CrowdPitch Pro</center>", unsafe_allow_html=True)
