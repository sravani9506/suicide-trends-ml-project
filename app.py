import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def find_file(name):
    for p in [BASE_DIR / name, BASE_DIR / "data" / name]:
        if p.exists():
            return p
    found = list(BASE_DIR.rglob(name))
    return found[0] if found else BASE_DIR / name

MODEL_PATH = find_file("model.pkl")
RECENT_PATH = find_file("NCRB_ADSI_2023_Table_2.1.csv")
OLD_PATH = find_file("Suicides in India 2001-2012.csv")

# Page Config
st.set_page_config(
    page_title="Mental Health & Suicide Trends Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #1e222d;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    h1, h2, h3 { color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_old_data():
    if not OLD_PATH.exists():
        return None
    return pd.read_csv(OLD_PATH)

@st.cache_data
def load_recent_data():
    df = pd.read_csv(RECENT_PATH)
    df = df.rename(columns={
        "Year (Col.2)": "Year",
        "Total Number of Suicides (Col.3)": "Total_Suicides",
        "Projected Mid-Year Population (In Lakh) (Col.4)": "Population_Lakh",
        "Rate of Suicides (Col.3/Col.4) (Col.5)": "Suicide_Rate"
    })
    return df.sort_values("Year")

# Sidebar
st.sidebar.title("📌 Navigation Menu")
page = st.sidebar.radio("Go to", ["🏠 Dashboard", "📊 Data Analysis", "🤖 ML Trend Forecast"])

recent = load_recent_data()
old = load_old_data()

if page == "🏠 Dashboard":
    st.title("📈 Suicide Trends & Mental Health Data Analytics")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Historical Records", f"{len(old):,}" if old is not None else "N/A")
    col2.metric("Annual Survey Records", len(recent))
    col3.metric("Latest Record Year", int(recent["Year"].max()))

    st.markdown("### 📊 Reported National Suicide Trends Over Time")
    fig = px.line(
        recent, x="Year", y="Total_Suicides", 
        markers=True, template="plotly_dark",
        title="Annual Reported Suicides in India"
    )
    fig.update_traces(line_color='#00d26a', line_width=3)
    st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Data Analysis":
    st.title("📊 Detailed Data Explorations")
    if old is not None:
        causes_df = old[old["Type_code"] == "Causes"]
        
        st.subheader("Top 10 Affected States (2001–2012)")
        state_data = causes_df.groupby("State")["Total"].sum().reset_index().sort_values("Total", ascending=False).head(10)
        fig_state = px.bar(state_data, x="Total", y="State", orientation='h', template="plotly_dark", color="Total", color_continuous_scale="Reds")
        st.plotly_chart(fig_state, use_container_width=True)
        
        col_g, col_a = st.columns(2)
        with col_g:
            st.subheader("Gender Breakdown")
            gender_df = causes_df.groupby("Gender")["Total"].sum().reset_index()
            fig_gender = px.pie(gender_df, values="Total", names="Gender", hole=0.4, template="plotly_dark")
            st.plotly_chart(fig_gender, use_container_width=True)
            
        with col_a:
            st.subheader("Age Group Breakdown")
            age_df = causes_df.groupby("Age_group")["Total"].sum().reset_index()
            fig_age = px.bar(age_df, x="Age_group", y="Total", template="plotly_dark", color="Total")
            st.plotly_chart(fig_age, use_container_width=True)
    else:
        st.error("Historical Dataset not found.")

elif page == "🤖 ML Trend Forecast":
    st.title("🤖 Future Trend Estimation (ML)")
    bundle = load_model()
    model = bundle["model"]
    
    selected_year = st.slider("Select Year for Prediction", 2013, 2030, 2025)
    if st.button("Generate Estimate"):
        prediction = model.predict(pd.DataFrame({"Year": [selected_year]}))[0]
        st.metric(f"Estimated Suicide Count for {selected_year}", f"{max(0, round(prediction)):,}")
        st.info("Note: This simple linear model uses historical data for trend forecasting.")
