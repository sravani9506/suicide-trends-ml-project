import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def find_file(name):
    for p in [BASE_DIR / name, BASE_DIR / "data" / name]:
        if p.exists():
            return p
    # recursive search
    found = list(BASE_DIR.rglob(name))
    return found[0] if found else BASE_DIR / name

MODEL_PATH = find_file("model.pkl")
RECENT_PATH = find_file("NCRB_ADSI_2023_Table_2.1.csv")
OLD_PATH = find_file("Suicides in India 2001-2012.csv")

st.set_page_config(page_title="Suicide Trends in India", page_icon="📊", layout="wide")

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

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Home","Dataset View","Year Wise Analysis","State Wise Analysis","Gender Analysis","Age Group Analysis","ML Demo"])

st.title("📊 Suicide Trends in India")
st.caption("Historical data analysis and an educational ML demonstration")

# [Nee migilina code antha same - just old_data None check add chesa]

if page == "Home":
    st.header("Project Overview")
    recent = load_recent_data()
    old = load_old_data()
    c1,c2,c3 = st.columns(3)
    c1.metric("Historical records", f"{len(old):,}" if old is not None else "Not uploaded")
    c2.metric("Annual records", len(recent))
    c3.metric("Latest year", int(recent["Year"].max()))
    fig, ax = plt.subplots()
    ax.plot(recent["Year"], recent["Total_Suicides"], marker="o")
    ax.set_xlabel("Year"); ax.set_ylabel("Reported suicide count")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)
    st.info("This is population-level historical analysis, not an individual risk prediction.")

elif page == "State Wise Analysis":
    st.header("State Wise Analysis (2001–2012)")
    df = load_old_data()
    if df is None:
        st.error("Old dataset not found in GitHub. Please upload 'Suicides in India 2001-2012.csv' to repo root.")
    else:
        df = df[df["Type_code"] == "Causes"]
        state_data = df.groupby("State", as_index=False)["Total"].sum().sort_values("Total", ascending=False)
        st.dataframe(state_data, use_container_width=True)
        fig, ax = plt.subplots()
        top_states = state_data.head(10).sort_values("Total")
        ax.barh(top_states["State"], top_states["Total"])
        st.pyplot(fig); plt.close(fig)
#... same pattern for Gender/Age...

elif page == "ML Demo":
    st.header("Linear Regression Demo")
    bundle = load_model()
    model = bundle["model"]
    year = st.number_input("Enter a year to explore", 2013, 2030, 2024)
    if st.button("Estimate reported count"):
        prediction = model.predict(pd.DataFrame({"Year": [year]}))[0]
        st.metric(f"Model estimate for {year}", f"{max(0, round(prediction)):,}")