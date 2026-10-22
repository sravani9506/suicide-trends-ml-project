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

# Page Configuration
st.set_page_config(
    page_title="MindMetrics | Health Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Real App Design
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border: 1px solid #30363d;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: #58a6ff;
    }
    div[data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-weight: 700 !important;
    }

    /* Custom Recommendation Cards */
    .app-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .app-card h4 {
        color: #58a6ff;
        margin-bottom: 12px;
    }
    .app-card p {
        color: #8b949e;
        font-size: 14px;
    }
    .app-card ul {
        margin-top: 8px;
        padding-left: 20px;
        color: #c9d1d9;
    }

    /* Buttons */
    .stButton>button {
        background-color: #238636;
        color: #ffffff;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #2ea043;
    }
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

# Navigation Menu
st.sidebar.markdown("### 🧠 MindMetrics AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "🏠 Overview Dashboard", 
    "📊 Trend Analytics", 
    "🤖 ML Forecasting", 
    "💡 Action Plans & Helpline"
])

recent = load_recent_data()
old = load_old_data()

# 1. OVERVIEW DASHBOARD
if page == "🏠 Overview Dashboard":
    st.title("📈 National Mental Health & Trend Analytics")
    st.markdown("Public Health Insights & Historical Data Intelligence")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Historical Cause Records", f"{len(old):,}" if old is not None else "N/A")
    col2.metric("Annual Survey Timeline", f"{len(recent)} Years")
    col3.metric("Latest Report Year", int(recent["Year"].max()))

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 National Reported Suicide Growth")
    fig = px.line(
        recent, x="Year", y="Total_Suicides", 
        markers=True, template="plotly_dark",
        labels={"Total_Suicides": "Reported Cases", "Year": "Year"}
    )
    fig.update_traces(line_color='#58a6ff', line_width=3, marker_size=8)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# 2. TREND ANALYTICS
elif page == "📊 Trend Analytics":
    st.title("📊 Demographic Data Explorations")
    st.markdown("---")
    if old is not None:
        causes_df = old[old["Type_code"] == "Causes"]
        
        st.subheader(" Top 10 High-Impact States (Historical Sum)")
        state_data = causes_df.groupby("State")["Total"].sum().reset_index().sort_values("Total", ascending=False).head(10)
        fig_state = px.bar(
            state_data, x="Total", y="State", orientation='h', 
            template="plotly_dark", color="Total", color_continuous_scale="Viridis"
        )
        fig_state.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_state, use_container_width=True)
        
        col_g, col_a = st.columns(2)
        with col_g:
            st.subheader("Gender Share")
            gender_df = causes_df.groupby("Gender")["Total"].sum().reset_index()
            fig_gender = px.pie(gender_df, values="Total", names="Gender", hole=0.5, template="plotly_dark", color_discrete_sequence=['#58a6ff', '#f778ba'])
            fig_gender.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gender, use_container_width=True)
            
        with col_a:
            st.subheader("Age Group Distribution")
            age_df = causes_df.groupby("Age_group")["Total"].sum().reset_index()
            fig_age = px.bar(age_df, x="Age_group", y="Total", template="plotly_dark", color_discrete_sequence=['#238636'])
            fig_age.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_age, use_container_width=True)
    else:
        st.error("Historical Dataset not found.")

# 3. ML FORECASTING
elif page == "🤖 ML Forecasting":
    st.title("🤖 ML Predictive Projection Model")
    st.markdown("Linear trend forecasting engine based on historical survey points")
    st.markdown("---")
    
    bundle = load_model()
    model = bundle["model"]
    
    selected_year = st.slider("Target Projection Year", 2013, 2030, 2026)
    
    if st.button("Run ML Trend Forecast"):
        prediction = model.predict(pd.DataFrame({"Year": [selected_year]}))[0]
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(f"Estimated Projection for {selected_year}", f"{max(0, round(prediction)):,} Cases")
        st.info("ℹ️ Educational Projection: Simple linear extrapolation trained on annual survey records.")

# 4. ACTION PLANS & HELPLINE
elif page == "💡 Action Plans & Helpline":
    st.title("💡 Strategic Intervention & Policy Recommendations")
    st.markdown("Data-driven intervention roadmaps and mental health awareness frameworks.")
    st.markdown("---")
    
    if old is not None:
        st.subheader("🎯 State-Level Action Plan Matrix")
        states_list = sorted(old["State"].unique())
        selected_state = st.selectbox("Choose Target State:", states_list)
        
        state_total = old[(old["State"] == selected_state) & (old["Type_code"] == "Causes")]["Total"].sum()
        
        st.markdown(f"""
        <div class="app-card">
            <h4>📍 Tailored Strategy: {selected_state}</h4>
            <p>Historical Total Recorded Cause Occurrences: <b>{state_total:,}</b></p>
            <ul>
                <li><b>Community Health Outreach:</b> Deploy mobile counseling units in high-density districts.</li>
                <li><b>Institutional Counseling:</b> Establish mandatory student mental health centers in universities.</li>
                <li><b>Tele-counseling Integration:</b> Sync state health helpline infrastructure with Tele-MANAS framework.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("🏛️ Primary Policy Directives")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("""
        <div class="app-card">
            <h4>👨‍🎓 Youth Focus Initiatives (Age 15-29)</h4>
            <ul>
                <li><b>Academic Support:</b> Early detection of exam-related distress & peer-support circles.</li>
                <li><b>Stigma Reduction:</b> Public awareness drives in educational institutions.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="app-card">
            <h4>🌾 Rural & Economic Support Frameworks</h4>
            <ul>
                <li><b>Agricultural Debt Counseling:</b> Support centers in farming-dependent regions.</li>
                <li><b>Grassroots Training:</b> Train ASHA workers for early psychological distress identification.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.subheader("📞 Emergency Helpline Directory")
    st.info("""
    - **Tele-MANAS (Govt of India):** 14416 / 1800 891 4416 (24/7 Toll-Free)
    - **KIRAN Mental Health Line:** 1800-599-0019
    - **Vandrevala Foundation Helpline:** +91 9999 666 555
    """)