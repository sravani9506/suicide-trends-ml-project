import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# FIX: Try both root and data folder
def find_file(filename):
    # check root, data/, and ./data
    for p in [BASE_DIR / filename, BASE_DIR / "data" / filename, BASE_DIR / filename.replace("data/","")]:
        if p.exists():
            return p
    # also search by name only
    for p in BASE_DIR.rglob(filename):
        return p
    return BASE_DIR / filename

MODEL_PATH = find_file("model.pkl")
RECENT_CSV_NAME = "NCRB_ADSI_2023_Table_2.1.csv"
OLD_CSV_NAME = "Suicides in India 2001-2012.csv"

st.set_page_config(page_title="Suicide Trends in India", page_icon="📊", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_old_data():
    path = find_file(OLD_CSV_NAME)
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data
def load_recent_data():
    path = find_file(RECENT_CSV_NAME)
    df = pd.read_csv(path)
    # rename for safety
    df = df.rename(columns={
        "Year (Col.2)": "Year",
        "Total Number of Suicides (Col.3)": "Total_Suicides",
        "Projected Mid-Year Population (In Lakh) (Col.4)": "Population_Lakh",
        "Rate of Suicides (Col.3/Col.4) (Col.5)": "Suicide_Rate"
    })
    return df.sort_values("Year")

# ... rest of your code same ...

# Replace your except block with this:
# except FileNotFoundError as e:
#     st.error(f"File not found: {e}")
#     st.info(f"Checked in {BASE_DIR}. Make sure {RECENT_CSV_NAME} and model.pkl are in GitHub root.")