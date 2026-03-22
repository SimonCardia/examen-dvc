import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json
import os

X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# Load trained model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R2:  {r2:.4f}")

# Save predictions
predictions = pd.DataFrame({"y_test": y_test.values, "y_pred": y_pred})
os.makedirs("data/processed", exist_ok=True)
predictions.to_csv("data/processed/predictions.csv", index=False)

# Save scores
os.makedirs("metrics", exist_ok=True)
scores = {"mse": mse, "r2": r2}
with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=4)

print("Predictions saved to data/processed/predictions.csv")
print("Scores saved to metrics/scores.json")
