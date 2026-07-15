import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv(r"D:\python\yield_prediction.csv")

# Show first 5 rows
print(df.head())

# Remove unwanted columns
df = df.drop(columns=['Unnamed: 13', 'Unnamed: 14'], errors='ignore')

# Convert categorical columns to numeric
le = LabelEncoder()

for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Features and target
X = df.drop("yield", axis=1)
y = df["yield"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
print("\nModel Performance")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Example prediction
sample = X.iloc[[0]]
prediction = model.predict(sample)

print("\nSample Prediction")
print("Predicted Yield:", prediction[0])

# -----------------------------
# Actual vs Predicted Scatter Plot
# -----------------------------
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)

# Perfect prediction reference line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2,
    label='Perfect Prediction'
)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Crop Yield")
plt.legend()
plt.grid(True)

plt.show()