import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# ------------------------
# Load Dataset
# ------------------------
df = pd.read_csv(r"D:\python\yield_prediction.csv")

df = df.drop(columns=['Unnamed: 13', 'Unnamed: 14'], errors='ignore')

# Encode categorical columns
le = LabelEncoder()

for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Features and Target
X = df.drop("yield", axis=1)
y = df["yield"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ------------------------
# Streamlit UI
# ------------------------

st.title("🌾 Smart Crop Yield Prediction")

st.write("Enter the crop details")

# Use first row as default values
sample = X.iloc[0]

inputs = {}

for column in X.columns:
    inputs[column] = st.number_input(
        column,
        value=float(sample[column])
    )

if st.button("Predict Yield"):

    input_df = pd.DataFrame([inputs])

    prediction = model.predict(input_df)

    st.success(f"🌾 Predicted Crop Yield: {prediction[0]:.2f}")