import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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

    # Models to compare
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    best_model = None
    best_r2 = -float("inf")
    metrics_data = {}

    print(f"{'Model':<20} | {'R2 Score':<10} | {'MAE':<10} | {'MSE':<10}")
    print("-" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        metrics_data[name] = {
            "R2": round(r2, 4),
            "MAE": round(mae, 2),
            "MSE": round(mse, 2)
        }

        print(f"{name:<20} | {r2:.4f}     | ${mae:.2f}    | {mse:.2f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    # Save best model
    joblib.dump(best_model, "cost_model.pkl")
    print(f"\nBest Model Saved: {best_model.__class__.__name__} (R2: {best_r2:.4f})")

    # Save metrics for frontend
    with open("model_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
    print("Metrics saved to model_metrics.json")

if __name__ == "__main__":
    train_model()
