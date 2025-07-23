import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("train_cleaned.csv")

df.dropna(inplace=True)

X = df.drop(columns=["selling_price"])
y = df["selling_price"]

for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("Model Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

model.fit(X, y)

import joblib
joblib.dump(model, "car_price_model.pkl")
