import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

def train_model():
    # Load dataset
    try:
        df = pd.read_csv("housing_data.csv")
    except FileNotFoundError:
        print("Error: housing_data.csv not found. Run generate_data.py first.")
        return

    # Features and Target
    X = df[['total_area', 'num_rooms', 'num_bedrooms', 'num_bathrooms', 'num_living_rooms', 'num_kitchens', 'quality_score', 'complexity_score']]
    y = df['estimated_cost']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Model Trained using Random Forest Regressor")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: ${mae:.2f}")

    # Save model
    joblib.dump(model, "cost_model.pkl")
    print("Model saved to cost_model.pkl")


if __name__ == "__main__":
    train_model()
