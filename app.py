
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="MindMetrics | India Trends",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Clean professional styling ----------
st.markdown("""
<style>
    :root {
        --navy: #14243A;
        --blue: #3768D8;
        --muted: #65758B;
        --line: #E3E9F1;
    }

    .stApp {
        background: #F5F7FB;
    }

    [data-testid="stHeader"] {
        background: rgba(245,247,251,0.92);
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--line);
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--navy) !important;
        letter-spacing: -0.4px;
    }

    p, label, .stMarkdown {
        color: #34445A;
    }

    .brand {
        font-size: 23px;
        font-weight: 750;
        color: var(--navy);
        letter-spacing: -0.7px;
    }

    .brand-sub {
        font-size: 12px;
        color: var(--muted);
        margin-top: 4px;
    }

    .hero {
        background: linear-gradient(115deg, #14243A, #233F68);
        padding: 30px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        color: #FFFFFF;
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: clamp(25px, 3vw, 36px);
        line-height: 1.2;
        margin: 0 0 10px 0;
        overflow-wrap: anywhere;
    }

    .hero p {
        color: #DCE7F7;
        font-size: 15px;
        margin: 0;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(20,36,58,0.035);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 13px;
    }

    div[data-testid="stMetricValue"] {
        color: var(--navy);
        font-size: clamp(22px, 2vw, 30px);
        font-weight: 700;
    }

    div[data-testid="stPlotlyChart"],
    div[data-testid="stDataFrame"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 8px;
    }

    .section-note {
        color: var(--muted);
        font-size: 14px;
        margin-top: -8px;
        margin-bottom: 20px;
    }

    .recommendation {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-left: 4px solid var(--blue);
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 12px;
    }

    .recommendation h4 {
        color: var(--navy);
        margin: 0 0 8px 0;
        font-size: 16px;
    }

    .recommendation p {
        color: #526277;
        margin: 0;
        line-height: 1.6;
    }

    .stButton > button {
        background: var(--blue);
        color: white;
        border: 0;
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #2853B5;
        color: white;
    }

    hr {
        border-color: var(--line);
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1rem;
        }

        .hero {
            padding: 22px 20px;
        }
    }
</style>
""", unsafe_allow_html=True)


# ---------- File handling ----------
def find_file(filename):
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "data" / filename
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


RECENT_PATH = find_file("NCRB_ADSI_2023_Table_2.1.csv")
OLD_PATH = find_file("Suicides in India 2001-2012.csv")
MODEL_PATH = find_file("model.pkl")


@st.cache_data
def load_recent_data(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Year (Col.2)": "Year",
        "Total Number of Suicides (Col.3)": "Total_Suicides",
        "Projected Mid-Year Population (In Lakh) (Col.4)": "Population_Lakh",
        "Rate of Suicides (Col.3/Col.4) (Col.5)": "Suicide_Rate"
    })

    required = [
        "Year", "Total_Suicides", "Population_Lakh", "Suicide_Rate"
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing annual dataset columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.dropna(subset=required)
        .sort_values("Year")
        .reset_index(drop=True)
    )


@st.cache_data
def load_old_data(path):
    return pd.read_csv(path)


def make_chart(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Arial", color="#34445A", size=12),
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hoverlabel=dict(bgcolor="#FFFFFF")
    )
    fig.update_xaxes(showgrid=False, linecolor="#E3E9F1")
    fig.update_yaxes(gridcolor="#EDF1F6", zeroline=False)
    return fig


def estimate_model(df):
    # Prefer the saved model; fallback if it cannot be loaded.
    if MODEL_PATH is not None:
        try:
            bundle = joblib.load(MODEL_PATH)
            model = bundle["model"] if isinstance(bundle, dict) else bundle
            return model, "Saved model"
        except Exception:
            pass

    model = LinearRegression()
    model.fit(df[["Year"]], df["Total_Suicides"])
    return model, "Model trained from available annual data"


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div class="brand">MindMetrics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">PUBLIC HEALTH ANALYTICS</div>',
        unsafe_allow_html=True
    )
    st.divider()
    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Trend Analytics",
            "ML Forecasting",
            "Action Plans & Helpline"
        ]
    )
    st.divider()
    st.caption("Academic Machine Learning Project")
    st.caption("Historical data analysis • India")


# ---------- Load data ----------
if RECENT_PATH is None:
    st.error("Annual CSV not found. Upload it to the repository root or data folder.")
    st.stop()

try:
    recent = load_recent_data(str(RECENT_PATH))
except Exception as exc:
    st.error(f"Unable to load annual dataset: {exc}")
    st.stop()

old = None
if OLD_PATH is not None:
    try:
        old = load_old_data(str(OLD_PATH))
    except Exception:
        old = None


# ---------- Overview ----------
if page == "Overview":
    st.markdown("""
    <div class="hero">
        <h1>India Suicide Trends &amp; Public Health Analytics</h1>
        <p>Historical insights, demographic exploration and an educational
        machine learning trend demonstration.</p>
    </div>
    """, unsafe_allow_html=True)

    latest = recent.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Annual observations", f"{len(recent)}")
    c2.metric("Latest report year", f"{int(latest['Year'])}")
    c3.metric("Latest reported count", f"{int(latest['Total_Suicides']):,}")
    c4.metric("Reported rate", f"{latest['Suicide_Rate']:.1f}")

    st.subheader("National reported trend")
    st.markdown(
        '<div class="section-note">Annual figures from the provided NCRB dataset.</div>',
        unsafe_allow_html=True
    )

    fig = px.line(
        recent, x="Year", y="Total_Suicides",
        markers=True,
        labels={
            "Year": "Year",
            "Total_Suicides": "Reported count"
        }
    )
    fig.update_traces(line_color="#3768D8", line_width=3, marker_size=7)
    st.plotly_chart(make_chart(fig), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Project objectives")
        st.markdown("""
        - Explore published historical data.
        - Visualize changes across years and demographic groups.
        - Demonstrate a basic Linear Regression model.
        - Support awareness and evidence-informed discussion.
        """)

    with right:
        st.subheader("Data scope")
        st.markdown("""
        - Annual national summary: 2013–2023.
        - Detailed demographic dataset: 2001–2012, if available.
        - Historical patterns do not establish causes or individual risk.
        """)

    st.info(
        "Educational project only. This dashboard does not assess an individual's "
        "mental health or predict individual suicide risk."
    )


# ---------- Trend analytics ----------
elif page == "Trend Analytics":
    st.markdown("""
    <div class="hero">
        <h1>Trend Analytics</h1>
        <p>Explore annual, state-level and demographic patterns in the available data.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["National trends", "Demographic analysis"])

    with tab1:
        left, right = st.columns(2)

        with left:
            st.subheader("Reported count by year")
            fig = px.line(
                recent, x="Year", y="Total_Suicides", markers=True,
                labels={"Total_Suicides": "Reported count", "Year": "Year"}
            )
            fig.update_traces(line_color="#3768D8", line_width=3)
            st.plotly_chart(make_chart(fig), use_container_width=True)

        with right:
            st.subheader("Reported rate by year")
            fig = px.line(
                recent, x="Year", y="Suicide_Rate", markers=True,
                labels={"Suicide_Rate": "Reported rate", "Year": "Year"}
            )
            fig.update_traces(line_color="#159A8C", line_width=3)
            st.plotly_chart(make_chart(fig), use_container_width=True)

        first, last = recent.iloc[0], recent.iloc[-1]
        change = (
            (last["Total_Suicides"] - first["Total_Suicides"])
            / first["Total_Suicides"] * 100
        )
        st.metric(
            f"Change in reported count ({int(first['Year'])}–{int(last['Year'])})",
            f"{change:+.2f}%"
        )
        st.caption("Change in reported counts alone does not explain why they changed.")

    with tab2:
        if old is None:
            st.warning(
                "The historical dataset is not available. Upload "
                "'Suicides in India 2001-2012.csv' to view demographic analysis."
            )
        elif not {"Type_code", "Total", "State", "Gender", "Age_group"}.issubset(old.columns):
            st.error("The historical dataset does not contain the expected columns.")
        else:
            causes = old[old["Type_code"] == "Causes"].copy()

            st.subheader("State-level historical category sums")
            state_data = (
                causes.groupby("State", as_index=False)["Total"]
                .sum().sort_values("Total", ascending=False).head(10)
            )
            fig = px.bar(
                state_data.sort_values("Total"),
                x="Total", y="State", orientation="h",
                labels={"Total": "Sum across cause categories", "State": "State"},
                color_discrete_sequence=["#3768D8"]
            )
            st.plotly_chart(make_chart(fig), use_container_width=True)

            left, right = st.columns(2)
            with left:
                st.subheader("Gender categories")
                gender_data = causes.groupby("Gender", as_index=False)["Total"].sum()
                fig = px.bar(
                    gender_data, x="Gender", y="Total",
                    labels={"Total": "Sum across cause categories"},
                    color_discrete_sequence=["#159A8C"]
                )
                st.plotly_chart(make_chart(fig), use_container_width=True)

            with right:
                st.subheader("Age-group categories")
                age_data = causes.groupby("Age_group", as_index=False)["Total"].sum()
                fig = px.bar(
                    age_data, x="Age_group", y="Total",
                    labels={"Total": "Sum across cause categories"},
                    color_discrete_sequence=["#7B68C4"]
                )
                st.plotly_chart(make_chart(fig), use_container_width=True)

            st.caption(
                "The Type_code='Causes' records can contain overlapping categories. "
                "These sums must not be interpreted as unique total suicides."
            )


# ---------- ML Forecasting ----------
elif page == "ML Forecasting":
    st.markdown("""
    <div class="hero">
        <h1>ML Forecasting</h1>
        <p>Interactive demonstration of a Linear Regression trend model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write(
        "Choose a future year to generate an experimental estimate from the "
        "annual reported-count trend."
    )

    try:
        model, model_source = estimate_model(recent)
        st.caption(f"Model source: {model_source}")

        last_year = int(recent["Year"].max())
        selected_year = st.slider(
            "Forecast year",
            min_value=last_year + 1,
            max_value=last_year + 10,
            value=last_year + 1
        )

        if st.button("Generate estimate"):
            prediction = float(
                model.predict(pd.DataFrame({"Year": [selected_year]}))[0]
            )
            st.metric(
                f"Experimental estimate for {selected_year}",
                f"{max(0, round(prediction)):,}"
            )
            st.warning(
                "This is a simple linear extrapolation based on only 11 annual "
                "observations. It may be inaccurate and is not an official forecast "
                "or an individual-level risk prediction."
            )

        st.subheader("Observed data")
        fig = px.scatter(
            recent, x="Year", y="Total_Suicides",
            labels={"Year": "Year", "Total_Suicides": "Reported count"}
        )
        fig.update_traces(marker=dict(size=9, color="#3768D8"))
        st.plotly_chart(make_chart(fig), use_container_width=True)

    except Exception as exc:
        st.error(f"Could not run the ML demonstration: {exc}")


# ---------- Action plans and helpline ----------
elif page == "Action Plans & Helpline":
    st.markdown("""
    <div class="hero">
        <h1>Awareness &amp; Action Planning</h1>
        <p>General public-health recommendations informed by the project theme.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Suggested awareness initiatives")

    st.markdown("""
    <div class="recommendation">
        <h4>Community outreach</h4>
        <p>Promote awareness programmes and make counselling information easier
        to access through local health services.</p>
    </div>
    <div class="recommendation">
        <h4>Student wellbeing</h4>
        <p>Encourage campus counselling access, peer support and early help-seeking
        without stigma or blame.</p>
    </div>
    <div class="recommendation">
        <h4>Rural and community support</h4>
        <p>Improve awareness of available professional support and strengthen
        referral pathways through local health services.</p>
    </div>
    """, unsafe_allow_html=True)

    if old is not None and "State" in old.columns:
        st.subheader("Explore historical state data")
        state = st.selectbox("Select a state", sorted(old["State"].dropna().unique()))
        if {"Type_code", "Total"}.issubset(old.columns):
            subset = old[
                (old["State"] == state) & (old["Type_code"] == "Causes")
            ]
            st.caption(
                f"{state}: {int(subset['Total'].sum()):,} summed cause-category "
                "records in the historical dataset; categories may overlap."
            )

    st.subheader("Mental health support in India")
    st.markdown("""
    **Tele-MANAS:** Call **14416** for mental health support in India.

    If someone is in immediate danger, contact local emergency services or
    seek urgent help from a trusted person or qualified professional.
    """)

st.divider()
st.caption(
    "MindMetrics | Academic project | Based on supplied historical datasets. "
    "The dashboard is for education and awareness, not diagnosis or individual prediction."
)
