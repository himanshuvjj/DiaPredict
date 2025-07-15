import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

df = pd.read_csv("data/diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
with open("model/diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved to model/diabetes_model.pkl")
