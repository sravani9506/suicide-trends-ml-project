
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
    page_title="MindMetrics | Suicide Prevention",
    page_icon="M",
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


# ---------------- PROFESSIONAL THEME ----------------
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
    background-color: #F1F5F9;
    background-image:
        linear-gradient(rgba(10, 29, 52, 0.84),
                        rgba(10, 29, 52, 0.90)),
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
    background: rgba(15, 35, 60, 0.94);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 26px;
    margin-bottom: 20px;
    box-shadow: 0 8px 26px rgba(0,0,0,0.15);
}}

.resource-card {{
    background: #FFFFFF;
    border: 1px solid #DCE5EF;
    border-left: 4px solid #168C9B;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
}}

.resource-card h4 {{
    color: #116879 !important;
    margin: 0 0 8px 0;
}}

.resource-card p {{
    color: #475569;
    line-height: 1.7;
}}

div[data-testid="stMetric"] {{
    background: #FFFFFF;
    border: 1px solid #DCE5EF;
    border-radius: 12px;
    padding: 17px;
}}

div[data-testid="stMetricLabel"] {{
    color: #52657A !important;
}}

div[data-testid="stMetricValue"] {{
    color: #123454 !important;
    font-weight: 700;
}}

.stButton > button {{
    border-radius: 9px;
    min-height: 43px;
    font-weight: 600;
    border: 1px solid #168C9B;
}}

.stButton > button[kind="primary"],
.stFormSubmitButton > button {{
    background: #087E8B;
    color: #FFFFFF;
}}

section[data-testid="stSidebar"] {{
    background: #10243D;
}}

section[data-testid="stSidebar"] * {{
    color: #F3F7FC;
}}

div[data-testid="stPlotlyChart"] {{
    background: #FFFFFF;
    border-radius: 12px;
    padding: 10px;
}}

hr {{
    border-color: rgba(255,255,255,0.2);
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
        raise ValueError(
            "Required columns not found: " + ", ".join(missing)
        )

    for col in [
        "Year", "Total_Suicides",
        "Population_Lakh", "Suicide_Rate"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            )

    return (
        df.dropna(subset=required)
        .sort_values("Year")
        .reset_index(drop=True)
    )


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

    # Fallback model for demonstration
    model = LinearRegression()
    model.fit(df[["Year"]], df["Total_Suicides"])
    return model


# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:
    st.markdown("<div style='height:6vh'></div>",
                unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("""
        <div style="padding:50px 10px 20px 8px;">
            <p style="color:#70D5D5;font-size:12px;
                      font-weight:700;letter-spacing:3px;">
                MINDMETRICS | PUBLIC HEALTH
            </p>
            <h1 style="font-size:46px;line-height:1.2;">
                Every Life<br>Matters.
            </h1>
            <p style="font-size:17px;line-height:1.9;
                      color:#E0EAF5;">
                Understanding trends. Encouraging awareness.
                Supporting emotional well-being.
            </p>
            <p style="font-size:14px;color:#B7C9DC;">
                Machine learning for suicide trend analysis
                and prevention awareness.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="glass-card">
            <p style="color:#70D5D5;font-size:12px;
                      font-weight:700;letter-spacing:2px;">
                USER ACCESS
            </p>
            <h2 style="margin-bottom:8px;">Welcome back</h2>
            <p style="color:#D1DDEB;font-size:14px;">
                Sign in to explore the dashboard and
                emotional wellness resources.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input(
                "Username or email",
                placeholder="Enter your username"
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password"
            )

            submitted = st.form_submit_button(
                "Sign in",
                use_container_width=True,
                type="primary"
            )

            if submitted:
                if username.strip() and password:
                    # Demo login only; credentials are not verified.
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error(
                        "Please enter your username and password."
                    )

        st.caption(
            "Academic demo only. This screen does not provide "
            "real account authentication."
        )

    st.stop()


# ---------------- LOAD DATA ----------------
recent = None
old = None

if RECENT_PATH is not None:
    try:
        recent = load_recent_data(str(RECENT_PATH))
    except Exception as error:
        st.error(f"Could not load the recent dataset: {error}")
else:
    st.warning(
        "Recent dataset not found. Add "
        "NCRB_ADSI_2023_Table_2.1.csv to the project or data folder."
    )

if OLD_PATH is not None:
    try:
        old = load_old_data(str(OLD_PATH))
    except Exception as error:
        st.warning(f"Could not load the historical dataset: {error}")


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## MINDMETRICS")
    st.caption("Prevention & Trend Analytics")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Prevention Resources",
            "Data Analysis",
            "ML Trend Prediction",
            "Emergency Support"
        ]
    )

    st.divider()
    st.caption("Academic project prototype")

    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()


# ---------------- PAGE HEADER HELPER ----------------
def page_header(title, subtitle):
    st.markdown(f"""
    <div class="glass-card">
        <h1 style="font-size:30px;margin-bottom:8px;">
            {title}
        </h1>
        <p style="color:#DCE7F3;margin-bottom:0;">
            {subtitle}
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- OVERVIEW ----------------
if page == "Overview":
    page_header(
        "Suicide Prevention & Trend Analytics",
        "A public health awareness prototype combining "
        "reported statistics, trend analysis and support resources."
    )

    if recent is not None and not recent.empty:
        c1, c2, c3 = st.columns(3)

        c1.metric("Years Available", len(recent))
        c2.metric("Latest Record Year", int(recent["Year"].max()))
        c3.metric(
            "Latest Reported Total",
            f"{int(recent.iloc[-1]['Total_Suicides']):,}"
        )

        st.subheader("National Trend")

        fig = px.line(
            recent,
            x="Year",
            y="Total_Suicides",
            markers=True,
            template="plotly_white",
            labels={
                "Year": "Year",
                "Total_Suicides": "Reported total"
            }
        )
        fig.update_traces(
            line_color="#087E8B",
            line_width=3,
            marker_size=7
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(family="Arial", color="#24364B")
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(
            "Add the NCRB dataset to enable statistics and charts."
        )

    st.markdown("""
    <div class="resource-card">
        <h4>Project purpose</h4>
        <p>
            Analyse historical public-health statistics and provide
            responsible emotional wellness resources. Aggregate
            trends cannot determine an individual's suicide risk.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- PREVENTION RESOURCES ----------------
elif page == "Prevention Resources":
    page_header(
        "Emotional Wellness Resources",
        "Simple exercises that may help during stressful moments."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="resource-card">
            <h4>Pause and breathe</h4>
            <p>Take a comfortable position and breathe slowly.
            Stop if the exercise feels uncomfortable.</p>
        </div>
        <div class="resource-card">
            <h4>Write down your thoughts</h4>
            <p>Write what is troubling you and identify one
            manageable next step.</p>
        </div>
        <div class="resource-card">
            <h4>Reconnect with someone</h4>
            <p>Consider contacting a trusted friend, family
            member or another supportive person.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="resource-card">
            <h4>Ground yourself</h4>
            <p>Notice the things you can see, hear and feel
            around you to reconnect with the present.</p>
        </div>
        <div class="resource-card">
            <h4>Take a small break</h4>
            <p>Drink some water, rest, or move to a place
            where you feel more comfortable.</p>
        </div>
        <div class="resource-card">
            <h4>Seek professional support</h4>
            <p>A mental-health professional can help you
            work through difficult feelings.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "These resources are not a diagnosis or a replacement "
        "for professional mental-health care."
    )

    st.subheader("What would help you right now?")

    feeling = st.selectbox(
        "Choose an optional topic",
        [
            "I want to feel calmer",
            "I feel lonely",
            "I feel overwhelmed",
            "I want to talk to someone"
        ]
    )

    suggestions = {
        "I want to feel calmer":
            "Try a comfortable grounding activity and reach out "
            "to someone supportive if you need company.",
        "I feel lonely":
            "Consider sending a short message to a trusted person "
            "and asking if they have time to talk.",
        "I feel overwhelmed":
            "Choose one small task, pause other demands where "
            "possible, and consider speaking to someone you trust.",
        "I want to talk to someone":
            "Contact a trusted person or a qualified counsellor. "
            "You do not need to explain everything at once."
    }

    st.write(suggestions[feeling])


# ---------------- DATA ANALYSIS ----------------
elif page == "Data Analysis":
    page_header(
        "Historical Data Analysis",
        "Explore reported national trends and historical state data."
    )

    if recent is not None and not recent.empty:
        st.subheader("National trend over time")

        fig = px.line(
            recent,
            x="Year",
            y="Total_Suicides",
            markers=True,
            template="plotly_white"
        )
        fig.update_traces(line_color="#087E8B", line_width=3)
        fig.update_layout(
            font=dict(color="#24364B"),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("The recent NCRB dataset is not available.")

    if (
        old is not None
        and {"State", "Total", "Type_code"}.issubset(old.columns)
    ):
        st.subheader("Reported historical totals by state")

        causes = old[old["Type_code"] == "Causes"]

        if causes.empty:
            causes = old

        state_data = (
            causes.groupby("State", as_index=False)["Total"]
            .sum()
            .sort_values("Total", ascending=False)
            .head(10)
        )

        fig_state = px.bar(
            state_data,
            x="Total",
            y="State",
            orientation="h",
            template="plotly_white",
            labels={"Total": "Reported total", "State": "State"}
        )
        fig_state.update_traces(marker_color="#168C9B")
        fig_state.update_layout(
            yaxis={"categoryorder": "total ascending"},
            font=dict(color="#24364B"),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_state, use_container_width=True)

    elif old is None:
        st.info(
            "Add Suicides in India 2001-2012.csv to enable "
            "historical state analysis."
        )


# ---------------- ML TREND PREDICTION ----------------
elif page == "ML Trend Prediction":
    page_header(
        "Machine Learning Trend Estimation",
        "Estimate aggregate national totals using a saved model "
        "or a simple Linear Regression fallback."
    )

    st.warning(
        "This is an educational estimate of aggregate reported "
        "totals, not a prediction of an individual's behaviour "
        "or suicide risk."
    )

    if recent is not None and not recent.empty:
        model = get_model(recent)
        last_year = int(recent["Year"].max())

        selected_year = st.slider(
            "Select target year",
            min_value=last_year + 1,
            max_value=last_year + 12,
            value=last_year + 3
        )

        if st.button(
            "Generate estimate",
            type="primary",
            use_container_width=True
        ):
            try:
                prediction = model.predict(
                    pd.DataFrame({"Year": [selected_year]})
                )[0]

                st.metric(
                    f"Model estimate for {selected_year}",
                    f"{max(0, round(float(prediction))):,}"
                )

                st.caption(
                    "Model estimates depend on the training data, "
                    "model assumptions and future changes. They "
                    "should not be treated as official forecasts."
                )

            except Exception as error:
                st.error(f"Prediction could not be generated: {error}")

    else:
        st.error(
            "Prediction requires a valid dataset containing "
            "Year and Total_Suicides columns."
        )


# ---------------- EMERGENCY SUPPORT ----------------
elif page == "Emergency Support":
    page_header(
        "Mental Health & Emergency Support",
        "If you or someone else may be in immediate danger, "
        "seek urgent help from local emergency services."
    )

    st.markdown("""
    <div class="resource-card">
        <h4>Tele-MANAS — India</h4>
        <p>National tele-mental-health support: <b>14416</b></p>
        <p>Confirm service availability through official "
        "government health resources.</p>
    </div>
    <div class="resource-card">
        <h4>Emergency services — India</h4>
        <p>For immediate danger, call <b>112</b> or go to the
        nearest emergency department.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "An app cannot replace emergency responders, a qualified "
        "clinician, or direct support from a trusted person."
    )
