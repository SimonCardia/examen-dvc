import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

# Load best parameters
with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

print(f"Training with parameters: {best_params}")

# Train model
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# Save trained model
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved to models/model.pkl")
