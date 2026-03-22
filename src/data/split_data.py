import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Download data
url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
df = pd.read_csv(url)

# Drop date columns
df = df.select_dtypes(include=["float64", "int64"])

# Save to data/raw
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/raw.csv", index=False)
print(f"Data loaded: {df.shape}")

# Split
X = df.drop(columns=["silica_concentrate"])
y = df["silica_concentrate"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save
os.makedirs("data/processed", exist_ok=True)
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Split complete:")
print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
