import base64
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression

# ---------------- CONFIGURATION ----------------
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="MindMetrics | Public Health Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- FILE LOCATOR ----------------
def find_file(filename):
    for path in (
        BASE_DIR / filename,
        BASE_DIR / "data" / filename
    ):
        if path.is_file():
            return path

    matches = list(BASE_DIR.rglob(filename))
    return matches[0] if matches else None


RECENT_PATH = find_file("NCRB_ADSI_2023_Table_2.1.csv")
OLD_PATH = find_file("Suicides in India 2001-2012.csv")
MODEL_PATH = find_file("model.pkl")

BG_PATH = next(
    (BASE_DIR / name for name in
     ["bg.jpg", "bg.jpeg", "bg.png"]
     if (BASE_DIR / name).is_file()),
    None
)


# ---------------- PROFESSIONAL THEME & CSS ----------------
def get_background():
    if BG_PATH is None:
        return "none"

    encoded = base64.b64encode(BG_PATH.read_bytes()).decode()
    extension = BG_PATH.suffix.lower()
    mime = "image/png" if extension == ".png" else "image/jpeg"

    return f'url("data:{mime};base64,{encoded}")'


background = get_background()

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {{
    background-color: #0F172A;
    background-image:
        linear-gradient(rgba(10, 25, 47, 0.88),
                        rgba(10, 25, 47, 0.94)),
        {background};
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Inter', sans-serif;
}}

[data-testid="stHeader"] {{
    background: transparent;
}}

h1, h2, h3 {{
    color: #FFFFFF;
    font-weight: 700;
    letter-spacing: -0.4px;
}}

p, label, input, button {{
    font-family: 'Inter', sans-serif;
}}

.glass-card {{
    background: rgba(15, 35, 60, 0.85);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}}

.resource-card {{
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 5px solid #0EA5E9;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}}

.resource-card h4 {{
    color: #0284C7 !important;
    margin: 0 0 8px 0;
}}

.resource-card p {{
    color: #334155;
    line-height: 1.7;
}}

div[data-testid="stMetric"] {{
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}}

div[data-testid="stMetricLabel"] {{
    color: #64748B !important;
    font-weight: 600;
}}

div[data-testid="stMetricValue"] {{
    color: #0F172A !important;
    font-weight: 700;
}}

/* Custom Professional Sidebar Styling */
section[data-testid="stSidebar"] {{
    background-color: #0B1329 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}}

section[data-testid="stSidebar"] * {{
    color: #E2E8F0 !important;
}}

/* Professional Radio Buttons Menu Styling */
div[data-testid="stRadio"] > div {{
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    padding: 8px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}}

div[data-testid="stRadio"] label {{
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 4px;
    transition: all 0.2s ease;
    cursor: pointer;
    font-weight: 500;
}}

div[data-testid="stRadio"] label:hover {{
    background: rgba(255, 255, 255, 0.08);
}}

div[data-testid="stPlotlyChart"] {{
    background: #FFFFFF;
    border-radius: 14px;
    padding: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}}

.stButton > button {{
    border-radius: 10px;
    min-height: 44px;
    font-weight: 600;
    border: 1px solid #0EA5E9;
}}

.stButton > button[kind="primary"],
.stFormSubmitButton > button {{
    background: #0284C7;
    color: #FFFFFF;
}}
</style>
""", unsafe_allow_html=True)


# ---------------- DATA LOADERS ----------------
@st.cache_data
def load_recent_data(path):
    df = pd.read_csv(path)

    df = df.rename(columns={
        "Year (Col.2)": "Year",
        "Total Number of Suicides (Col.3)": "Total_Suicides",
        "Projected Mid-Year Population (In Lakh) (Col.4)": "Population_Lakh",
        "Rate of Suicides (Col.3/Col.4) (Col.5)": "Suicide_Rate"
    })

    required = ["Year", "Total_Suicides"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError("Required columns not found: " + ", ".join(missing))

    for col in ["Year", "Total_Suicides", "Population_Lakh", "Suicide_Rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            )

    return df.dropna(subset=required).sort_values("Year").reset_index(drop=True)


@st.cache_data
def load_old_data(path):
    return pd.read_csv(path)


def get_model(df):
    if MODEL_PATH is not None:
        try:
            saved = joblib.load(MODEL_PATH)
            if isinstance(saved, dict):
                saved = saved.get("model")
            if saved is not None and hasattr(saved, "predict"):
                return saved
        except Exception:
            pass

    model = LinearRegression()
    model.fit(df[["Year"]], df["Total_Suicides"])
    return model


# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:
    st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("""
        <div style="padding:40px 10px 20px 8px;">
            <p style="color:#38BDF8;font-size:13px;font-weight:700;letter-spacing:3px;">
                MINDMETRICS | ANALYTICS & PREVENTION
            </p>
            <h1 style="font-size:48px;line-height:1.2;">
                Every Life<br>Matters.
            </h1>
            <p style="font-size:17px;line-height:1.8;color:#94A3B8;">
                Analyzing historical patterns, spreading mental health awareness,
                and offering evidence-based coping mechanisms.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="glass-card">
            <h2 style="margin-bottom:6px;font-size:24px;">Welcome Back</h2>
            <p style="color:#94A3B8;font-size:14px;">Sign in to access analytics and crisis resources.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

            if submitted:
                if username.strip() and password:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Please provide valid credentials.")

        st.caption("Academic Demo System • Unauthorized access prohibited")

    st.stop()


# ---------------- LOAD DATA ----------------
recent = None
old = None

if RECENT_PATH is not None:
    try:
        recent = load_recent_data(str(RECENT_PATH))
    except Exception as error:
        st.error(f"Could not load recent dataset: {error}")

if OLD_PATH is not None:
    try:
        old = load_old_data(str(OLD_PATH))
    except Exception as error:
        st.warning(f"Could not load historical dataset: {error}")


# ---------------- PROFESSIONAL SIDEBAR NAVIGATION ----------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0px 20px 0px;">
        <h2 style="font-size: 22px; margin: 0; color: #38BDF8 !important;">MINDMETRICS</h2>
        <p style="font-size: 12px; color: #64748B !important; margin: 0;">Public Health Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "MAIN MENU",
        [
            "🏠 Overview Dashboard",
            "📊 Data & Trend Analysis",
            "🔮 ML Risk & Trend Estimation",
            "🌱 Prevention & Wellness",
            "🚨 Emergency Crisis Support"
        ],
        label_visibility="visible"
    )

    st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)
    st.divider()
    st.caption("Academic Project Prototype v2.1")

    if st.button("🚪 Logout System", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()


# ---------------- PAGE HEADER HELPER ----------------
def page_header(title, subtitle):
    st.markdown(f"""
    <div class="glass-card">
        <h1 style="font-size:28px;margin-bottom:6px;">{title}</h1>
        <p style="color:#94A3B8;margin-bottom:0;font-size:15px;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- OVERVIEW ----------------
if page == "🏠 Overview Dashboard":
    page_header(
        "Suicide Analytics & Prevention Dashboard",
        "A holistic analytical platform integrating national statistics, ML estimation, and mental wellness support."
    )

    if recent is not None and not recent.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Historical Years Tracked", len(recent))
        c2.metric("Latest Data Record", int(recent["Year"].max()))
        c3.metric("Latest Annual Reported Total", f"{int(recent.iloc[-1]['Total_Suicides']):,}")

        st.subheader("National Annual Reported Trend")
        fig = px.line(
            recent, x="Year", y="Total_Suicides", markers=True,
            template="plotly_white", labels={"Total_Suicides": "Reported Cases", "Year": "Year"}
        )
        fig.update_traces(line_color="#0284C7", line_width=3, marker_size=7)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ---------------- DATA ANALYSIS (WITH AGE GROUP & GENDER) ----------------
elif page == "📊 Data & Trend Analysis":
    page_header(
        "Comprehensive Historical Data Analysis",
        "Detailed breakdowns by Age Group, Gender Trajectories, Causes, and Regional Impact."
    )

    if recent is not None and not recent.empty:
        st.subheader("1. National Trajectory Over Time")
        fig_year = px.line(
            recent, x="Year", y="Total_Suicides", markers=True,
            template="plotly_white", labels={"Total_Suicides": "Total Reported Cases"}
        )
        fig_year.update_traces(line_color="#0284C7", line_width=3)
        st.plotly_chart(fig_year, use_container_width=True)

    if old is not None and not old.empty:
        # --- AGE GROUP ANALYSIS ---
        st.subheader("2. Age Group Analysis")
        if "Age_group" in old.columns:
            # Filter out generic or '0-100+' totals if present
            age_df = old[~old["Age_group"].isin(["0-100+", "Total"])].groupby("Age_group", as_index=False)["Total"].sum()
            fig_age = px.bar(
                age_df, x="Age_group", y="Total",
                template="plotly_white",
                color="Total",
                color_continuous_scale="Blues",
                labels={"Age_group": "Age Bracket", "Total": "Total Cases"}
            )
            fig_age.update_layout(xaxis_title="Age Group", yaxis_title="Reported Cases")
            st.plotly_chart(fig_age, use_container_width=True)

        # --- GENDER ANALYSIS ---
        st.subheader("3. Gender Distribution Over Time")
        if "Gender" in old.columns and "Year" in old.columns:
            gender_year = old.groupby(["Year", "Gender"], as_index=False)["Total"].sum()
            fig_gender = px.bar(
                gender_year, x="Year", y="Total", color="Gender",
                barmode="group", template="plotly_white",
                color_discrete_map={"Male": "#0284C7", "Female": "#F43F5E"}
            )
            st.plotly_chart(fig_gender, use_container_width=True)

        # --- CAUSES BREAKDOWN ---
        st.subheader("4. Top Reported Causes Breakdown")
        if "Type_code" in old.columns and "Type" in old.columns:
            causes_df = old[old["Type_code"] == "Causes"]
            if not causes_df.empty:
                top_causes = causes_df.groupby("Type", as_index=False)["Total"].sum().sort_values("Total", ascending=False).head(10)
                fig_causes = px.bar(
                    top_causes, x="Total", y="Type", orientation="h",
                    template="plotly_white", labels={"Type": "Cause", "Total": "Total Cases"}
                )
                fig_causes.update_traces(marker_color="#0EA5E9")
                fig_causes.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_causes, use_container_width=True)

        # --- STATE-WISE BREAKDOWN ---
        st.subheader("5. Top 10 Most Affected States")
        if "State" in old.columns:
            state_df = old.groupby("State", as_index=False)["Total"].sum().sort_values("Total", ascending=False).head(10)
            fig_state = px.bar(
                state_df, x="State", y="Total", template="plotly_white",
                color="Total", color_continuous_scale="Tealgrn"
            )
            st.plotly_chart(fig_state, use_container_width=True)


# ---------------- ML TREND PREDICTION ----------------
elif page == "🔮 ML Risk & Trend Estimation":
    page_header(
        "Machine Learning Trend Estimation & Risk Drivers",
        "Statistical projections combined with socio-economic and psychological risk factor evaluations."
    )

    if recent is not None and not recent.empty:
        model = get_model(recent)
        last_year = int(recent["Year"].max())

        selected_year = st.slider(
            "Select Target Estimation Year",
            min_value=last_year + 1,
            max_value=last_year + 12,
            value=last_year + 3
        )

        if st.button("Generate ML Estimate & Analysis", type="primary", use_container_width=True):
            try:
                prediction = model.predict(pd.DataFrame({"Year": [selected_year]}))[0]
                pred_val = max(0, round(float(prediction)))

                st.metric(f"ML Projected Estimate for Year {selected_year}", f"{pred_val:,}")

                st.divider()
                st.subheader("Key Contributing Drivers behind High Projections")

                reasons_data = {
                    "Risk Category": [
                        "Family Problems & Interpersonal Conflicts",
                        "Financial Distress & Debt Load",
                        "Chronic Physical Illnesses",
                        "Clinical Depression & Mental Health Stigma",
                        "Career Pressure & Academic Failure",
                        "Substance Addiction & Dependency"
                    ],
                    "Estimated Share (%)": ["30 - 34%", "18 - 22%", "18 - 20%", "10 - 15%", "6 - 8%", "5 - 7%"],
                    "Primary Socio-Economic Factors": [
                        "Marital issues, domestic disputes, loneliness, and weak social support networks.",
                        "Sudden crop failure, business loss, unbearable debt burdens, and inflation shocks.",
                        "Long-term severe health complications, chronic pain, and limited healthcare access.",
                        "Lack of awareness, social stigma around therapy, and untreated severe depression.",
                        "Unemployment, fierce competitive exam failure, and high youth career anxiety.",
                        "Alcoholism, drug abuse, and resulting emotional breakdown."
                    ]
                }
                st.table(pd.DataFrame(reasons_data))

            except Exception as error:
                st.error(f"Estimation error: {error}")


# ---------------- PREVENTION RESOURCES ----------------
elif page == "🌱 Prevention & Wellness":
    page_header(
        "Self-Help Coping Strategies & Emotional Resources",
        "Actionable steps tailored to your current emotional state."
    )

    feeling = st.selectbox(
        "How are you feeling right now? Select an option:",
        [
            "Select an option...",
            "I want to feel calmer",
            "I feel lonely",
            "I feel overwhelmed",
            "I want to talk to someone"
        ]
    )

    if feeling == "I want to feel calmer":
        st.markdown("""
        <div class="resource-card">
            <h4>1. 4-7-8 Breathing Technique</h4>
            <p>Breathe in quietly through your nose for 4 seconds, hold for 7 seconds, and exhale completely through your mouth for 8 seconds. Repeat 4 times.</p>
        </div>
        <div class="resource-card">
            <h4>2. Progressive Muscle Relaxation</h4>
            <p>Tense your toes for 5 seconds, then release. Work your way up through legs, stomach, shoulders, and jaw.</p>
        </div>
        """, unsafe_allow_html=True)

    elif feeling == "I feel lonely":
        st.markdown("""
        <div class="resource-card">
            <h4>1. Micro-Connections</h4>
            <p>Send a quick friendly message or meme to an acquaintance or old friend. Small connections help break isolation.</p>
        </div>
        <div class="resource-card">
            <h4>2. Shared Environment</h4>
            <p>Spend time in active public settings like libraries, cafes, or parks to feel connected with surrounding society.</p>
        </div>
        """, unsafe_allow_html=True)

    elif feeling == "I feel overwhelmed":
        st.markdown("""
        <div class="resource-card">
            <h4>1. 5-4-3-2-1 Grounding Technique</h4>
            <p>Identify <b>5</b> things you see, <b>4</b> you touch, <b>3</b> you hear, <b>2</b> you smell, and <b>1</b> you taste.</p>
        </div>
        <div class="resource-card">
            <h4>2. Single Micro-Tasking</h4>
            <p>Write down everything overwhelming you. Pick ONE tiny task taking under 2 minutes, complete it, and pause the rest.</p>
        </div>
        """, unsafe_allow_html=True)

    elif feeling == "I want to talk to someone":
        st.markdown("""
        <div class="resource-card">
            <h4>1. Reach Out to Loved Ones</h4>
            <p>Call or text a trusted family member or colleague. Simply stating 'I am having a tough day' is a great starting step.</p>
        </div>
        <div class="resource-card">
            <h4>2. Verified Free Helplines</h4>
            <p>Call Tele-MANAS at <b>14416</b> or KIRAN at <b>1800-599-0019</b> for 24/7 free, confidential counseling.</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------- EMERGENCY SUPPORT ----------------
elif page == "🚨 Emergency Crisis Support":
    page_header(
        "24/7 National Emergency Helplines",
        "If you or someone you know is in distress or danger, contact emergency services immediately."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="resource-card" style="border-left: 5px solid #EF4444;">
            <h4>🚨 Tele-MANAS (Government of India)</h4>
            <p><b>Toll-Free Number:</b> 14416 or 1800-891-4416</p>
            <p><b>Availability:</b> 24/7 | Multi-Lingual Mental Health Support</p>
        </div>
        <div class="resource-card" style="border-left: 5px solid #EF4444;">
            <h4>📞 KIRAN Helpline</h4>
            <p><b>Toll-Free Number:</b> 1800-599-0019</p>
            <p><b>Availability:</b> 24/7 National Rehabilitation & Support</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="resource-card" style="border-left: 5px solid #0284C7;">
            <h4>🤝 Vandrevala Foundation Helpline</h4>
            <p><b>Direct Line:</b> +91 9999 666 555</p>
            <p><b>Availability:</b> 24/7 Free Crisis Counseling</p>
        </div>
        <div class="resource-card" style="border-left: 5px solid #0284C7;">
            <h4>🚑 National Immediate Response</h4>
            <p><b>Emergency Unified Response:</b> 112</p>
            <p><b>Ambulance Services:</b> 102 / 108</p>
        </div>
        """, unsafe_allow_html=True)