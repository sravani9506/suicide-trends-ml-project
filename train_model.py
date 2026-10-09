
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression

# Step 1: Set project paths
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "NCRB_ADSI_2023_Table_2.1.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

print("Step 1: Loading dataset...")

# Step 2: Load the CSV file
df = pd.read_csv(DATA_PATH)

# Step 3: Rename columns
df = df.rename(columns={
    "Year (Col.2)": "Year",
    "Total Number of Suicides (Col.3)": "Total_Suicides",
    "Projected Mid-Year Population (In Lakh) (Col.4)": "Population_Lakh",
    "Rate of Suicides (Col.3/Col.4) (Col.5)": "Suicide_Rate"
})

# Step 4: Select required columns
df = df[["Year", "Total_Suicides"]].dropna()
df = df.sort_values("Year")

print("Step 2: Dataset loaded!")
print("Number of rows:", len(df))
print(df.head())

# Step 5: Prepare input and target
X = df[["Year"]]
y = df["Total_Suicides"]

# Step 6: Create and train the model
print("Step 3: Training Linear Regression...")

model = LinearRegression()
model.fit(X, y)

# Step 7: Save the trained model
model_info = {
    "model": model,
    "training_year_min": int(df["Year"].min()),
    "training_year_max": int(df["Year"].max()),
    "target": "Total_Suicides",
    "model_type": "Linear Regression"
}

joblib.dump(model_info, MODEL_PATH)

print("Step 4: Model saved successfully!")
print("Model path:", MODEL_PATH)
print("Training years:", df["Year"].min(), "-", df["Year"].max())
print("Training rows:", len(df))