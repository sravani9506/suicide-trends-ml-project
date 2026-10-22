import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# ---------------- CONFIGURATION ----------------
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="MindMetrics | ML Public Health Intelligence",
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
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
}}

div[data-testid="stMetricLabel"] {{
    color: #1E293B !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}}

div[data-testid="stMetricValue"] {{
    color: #0284C7 !important;
    font-weight: 800 !important;
    font-size: 32px !important;
}}

section[data-testid="stSidebar"] {{
    background-color: #0B1329 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}}

section[data-testid="stSidebar"] * {{
    color: #E2E8F0 !important;
}}

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


# ---------------- ML TRAINER & EVALUATOR ----------------
def train_and_eval_models(df):
    X = df[["Year"]]
    y = df["Total_Suicides"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X)
        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)
        results[name] = {"model": model, "r2": r2, "mae": mae, "preds": preds}

    return results


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
                MINDMETRICS | ML ANALYTICS & PREVENTION
            </p>
            <h1 style="font-size:48px;line-height:1.2;">
                Every Life<br>Matters.
            </h1>
            <p style="font-size:17px;line-height:1.8;color:#94A3B8;">
                Analyzing historical patterns with Machine Learning algorithms,
                predicting trends, and offering structured crisis resources.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="glass-card">
            <h2 style="margin-bottom:6px;font-size:24px;">Welcome Back</h2>
            <p style="color:#94A3B8;font-size:14px;">Sign in to access ML analytics dashboard.</p>
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

        st.caption("Academic Demo System • ML Intelligence Prototype")

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


# ---------------- SIDEBAR NAVIGATION ----------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0px 20px 0px;">
        <h2 style="font-size: 22px; margin: 0; color: #38BDF8 !important;">MINDMETRICS</h2>
        <p style="font-size: 12px; color: #64748B !important; margin: 0;">Machine Learning Platform</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "MAIN MENU",
        [
            "🏠 Overview Dashboard",
            "📊 Data & Trend Analysis",
            "🤖 Machine Learning Center",
            "🌱 Prevention & Wellness",
            "🚨 Emergency Crisis Support"
        ],
        label_visibility="visible"
    )

    st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("""
    <div style='padding: 5px 0px;'>
        <p style='font-size: 13px; font-weight: 700; color: #38BDF8 !important; margin: 0;'>Academic Project Prototype</p>
        <p style='font-size: 11px; color: #94A3B8 !important; margin: 0;'>Machine Learning Project</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Logout", use_container_width=True):
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
        "A holistic platform integrating national data, machine learning projections, and mental health support."
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


# ---------------- DATA ANALYSIS ----------------
elif page == "📊 Data & Trend Analysis":
    page_header(
        "Comprehensive Historical Data Analysis",
        "Detailed breakdowns by Top 10 States, Age Groups, Gender Trajectories, and Causes."
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
        st.subheader("2. Top 10 States by Sum of Recorded Counts")
        if "State" in old.columns:
            causes_df = old[old["Type_code"] == "Causes"] if "Type_code" in old.columns else old
            state_df = causes_df.groupby("State", as_index=False)["Total"].sum().sort_values("Total", ascending=False).head(10)
            
            fig_state = px.bar(
                state_df, x="Total", y="State", orientation="h",
                template="plotly_white", labels={"Total": "Sum of recorded counts", "State": "State"},
                text_auto='.2s'
            )
            fig_state.update_traces(marker_color="#0284C7")
            fig_state.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_state, use_container_width=True)

        st.subheader("3. Age Group Analysis")
        if "Age_group" in old.columns:
            age_df = old[~old["Age_group"].isin(["0-100+", "Total"])].groupby("Age_group", as_index=False)["Total"].sum()
            fig_age = px.bar(
                age_df, x="Age_group", y="Total",
                template="plotly_white",
                color="Total",
                color_continuous_scale="Blues",
                labels={"Age_group": "Age Bracket", "Total": "Total Cases"}
            )
            st.plotly_chart(fig_age, use_container_width=True)

        st.subheader("4. Gender Distribution Over Time")
        if "Gender" in old.columns and "Year" in old.columns:
            gender_year = old.groupby(["Year", "Gender"], as_index=False)["Total"].sum()
            fig_gender = px.bar(
                gender_year, x="Year", y="Total", color="Gender",
                barmode="group", template="plotly_white",
                color_discrete_map={"Male": "#0284C7", "Female": "#F43F5E"}
            )
            st.plotly_chart(fig_gender, use_container_width=True)

        st.subheader("5. Top Reported Causes Breakdown")
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


# ---------------- MACHINE LEARNING CENTER ----------------
elif page == "🤖 Machine Learning Center":
    page_header(
        "Machine Learning Model Evaluation & Trend Estimation",
        "Evaluate regression models (R² Score) and predict future public health trends."
    )

    if recent is not None and not recent.empty:
        ml_results = train_and_eval_models(recent)

        st.subheader("1. Model Performance ($R^2$ Score Accuracy)")
        st.caption("R² Score represents model fit accuracy (values closer to 1.0000 indicate higher accuracy).")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        
        lr_r2 = ml_results["Linear Regression"]["r2"]
        rf_r2 = ml_results["Random Forest Regressor"]["r2"]
        dt_r2 = ml_results["Decision Tree Regressor"]["r2"]

        m_col1.metric("Linear Regression R²", f"{lr_r2:.4f}")
        m_col2.metric("Random Forest R²", f"{rf_r2:.4f}")
        m_col3.metric("Decision Tree R²", f"{dt_r2:.4f}")

        st.divider()

        st.subheader("2. Select Algorithm for Trend Estimation")
        selected_model_name = st.selectbox(
            "Choose ML Model for Prediction:",
            list(ml_results.keys())
        )

        chosen_model_info = ml_results[selected_model_name]
        chosen_model = chosen_model_info["model"]

        last_year = int(recent["Year"].max())
        selected_year = st.slider(
            "Select Target Estimation Year",
            min_value=last_year + 1,
            max_value=last_year + 12,
            value=last_year + 3
        )

        if st.button("Run ML Prediction", type="primary", use_container_width=True):
            prediction = chosen_model.predict(pd.DataFrame({"Year": [selected_year]}))[0]
            pred_val = max(0, round(float(prediction)))

            st.metric(
                f"ML Estimate for Year {selected_year} ({selected_model_name})",
                f"{pred_val:,}"
            )

            st.divider()
            
            st.subheader(f"3. Algorithmic Behavior ({selected_model_name})")
            
            if selected_model_name == "Linear Regression":
                st.info("""
                **Linear Regression Model:**
                - Fits a straight line ($y = mx + c$) through the historical dataset.
                - Best suited for capturing steady, long-term upward or downward macro trends over time.
                """)
            elif selected_model_name == "Random Forest Regressor":
                st.success("""
                **Random Forest Regressor Model:**
                - An ensemble algorithm that averages predictions across multiple Decision Trees.
                - Provides high stability and smooth estimates without overfitting to minor fluctuations.
                """)
            elif selected_model_name == "Decision Tree Regressor":
                st.warning("""
                **Decision Tree Regressor Model:**
                - Learns rule-based splits based on threshold conditions in the historical data.
                - While achieving high accuracy on known data, it predicts constant step values beyond historical bounds.
                """)

            st.subheader("4. Key Socio-Economic Factors Driving Projections")

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
                    "Marital conflicts, domestic disputes, social isolation, and lack of family support systems.",
                    "Unexpected crop failure, business losses, heavy debt burdens, and economic inflation.",
                    "Long-term debilitating health conditions, chronic severe pain, and limited access to healthcare.",
                    "Lack of public awareness, societal stigma around therapy, and undiagnosed clinical depression.",
                    "Unemployment, intense academic competition, exam failure, and career uncertainty.",
                    "Alcohol dependency, substance abuse disorders, and emotional instability."
                ]
            }
            st.table(pd.DataFrame(reasons_data))


# ---------------- PREVENTION RESOURCES ----------------
elif page == "🌱 Prevention & Wellness":
    page_header(
        "Self-Help Coping Strategies & Emotional Resources",
        "Actionable, step-by-step guidance tailored to your current emotional state."
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
            <h4>1. 4-7-8 Deep Breathing Technique</h4>
            <p>Inhale quietly through your nose for <b>4 seconds</b>, hold your breath for <b>7 seconds</b>, and exhale completely through your mouth for <b>8 seconds</b>. Repeat this cycle 4 times.</p>
        </div>
        <div class="resource-card">
            <h4>2. Progressive Muscle Relaxation</h4>
            <p>Tense the muscles in your feet for <b>5 seconds</b>, then release completely. Slowly move upward through your legs, abdomen, shoulders, and jaw, tensing and relaxing each group.</p>
        </div>
        <div class="resource-card">
            <h4>3. Cold Water Splash & Grounding</h4>
            <p>Splash cool water on your face. This activates the mammalian dive response, helping slow down your heart rate and quickly calm your nervous system.</p>
        </div>
        """, unsafe_allow_html=True)

    elif feeling == "I feel lonely":
        st.markdown("""
        <div class="resource-card">
            <h4>1. Initiate Micro-Connections</h4>
            <p>Send a quick text message or share a friendly post with a trusted friend or family member. Even small social interactions help break feelings of isolation.</p>
        </div>
        <div class="resource-card">
            <h4>2. Change Your Physical Environment</h4>
            <p>Visit a public space like a local cafe, library, or community park. Simply being in the presence of others can ease feelings of loneliness.</p>
        </div>
        <div class="resource-card">
            <h4>3. Join Interest-Based Groups</h4>
            <p>Engage with local or online groups focused on hobbies you enjoy (such as reading, art, fitness, or gaming) to connect with like-minded individuals.</p>
        </div>
        """, unsafe_allow_html=True)

    elif feeling == "I feel overwhelmed":
        st.markdown("""
        <div class="resource-card">
            <h4>1. 5-4-3-2-1 Grounding Method</h4>
            <p>Identify <b>5</b> things you can see, <b>4</b> things you can touch, <b>3</b> things you can hear, <b>2</b> things you can smell, and <b>1</b> thing you can taste around you.</p>
        </div>
        <div class="resource-card">
            <h4>2. Brain Dump & 2-Minute Micro Task</h4>
            <p>Write down everything causing you stress on paper. Select just ONE small task that takes under <b>2 minutes</b> to complete, do it, and pause the rest.</p>
        </div>
        <div class="resource-card">
            <h4>3. Pause Non-Essential Decisions</h4>
            <p>Step back from non-urgent decisions. Turn off unnecessary notifications for 2 hours and allow your mind time to rest and reset.</p>
        </div>
        """, unsafe_allow_html=True)

    elif feeling == "I want to talk to someone":
        st.markdown("""
        <div class="resource-card">
            <h4>1. Reach Out to Your Trusted Circle</h4>
            <p>Contact a trusted friend, family member, or colleague and say: <i>"I am having a tough day, can we talk?"</i> You do not need to explain everything at once.</p>
        </div>
        <div class="resource-card">
            <h4>2. Tele-MANAS Government Helpline (14416)</h4>
            <p>A 24/7 free mental health counseling service provided by the Government of India. Call <b>14416</b> or <b>1800-891-4416</b> for confidential support.</p>
        </div>
        <div class="resource-card">
            <h4>3. KIRAN Mental Health Support (1800-599-0019)</h4>
            <p>A national helpline managed by the Ministry of Social Justice and Empowerment offering free, professional counseling in 13 regional languages.</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------- EMERGENCY SUPPORT ----------------
elif page == "🚨 Emergency Crisis Support":
    page_header(
        "24/7 National Emergency Helplines",
        "If you or someone you know is in distress or immediate danger, contact emergency services right away."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="resource-card" style="border-left: 5px solid #EF4444;">
            <h4>🚨 Tele-MANAS (Government of India)</h4>
            <p><b>Toll-Free Helpline:</b> 14416 or 1800-891-4416</p>
            <p><b>Availability:</b> 24/7 | Free & Confidential Mental Health Counseling</p>
        </div>
        <div class="resource-card" style="border-left: 5px solid #EF4444;">
            <h4>📞 KIRAN Mental Health Helpline</h4>
            <p><b>Toll-Free Helpline:</b> 1800-599-0019</p>
            <p><b>Availability:</b> 24/7 | National Mental Health Rehabilitation Support</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="resource-card" style="border-left: 5px solid #0284C7;">
            <h4>🤝 Vandrevala Foundation Helpline</h4>
            <p><b>Direct Contact:</b> +91 9999 666 555</p>
            <p><b>Availability:</b> 24/7 Free Crisis Counseling</p>
        </div>
        <div class="resource-card" style="border-left: 5px solid #0284C7;">
            <h4>🚑 Immediate Emergency Response</h4>
            <p><b>Unified Emergency Number:</b> 112</p>
            <p><b>Ambulance Services:</b> 102 / 108</p>
        </div>
        """, unsafe_allow_html=True)