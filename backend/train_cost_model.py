import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "backend", "data", "house_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "cost_model.pkl")
CHART_PATH = os.path.join(BASE_DIR, "model_comparison.png")

def generate_data(n_samples=500):
    # (Same generation logic, keeping it brief for brevity if unchanged, but I'll include it to be safe)
    print("Generating synthetic housing data...")
    np.random.seed(42)
    areas = np.random.randint(30, 300, n_samples)
    rooms = []
    for a in areas:
        r = max(1, a // np.random.randint(25, 45))
        rooms.append(r)
    rooms = np.array(rooms)
    tiers = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
    base_rate = 2500 
    costs = []
    for i in range(n_samples):
        rate = base_rate
        if tiers[i] == 1: rate *= 1.5
        if tiers[i] == 2: rate *= 2.5
        noise = np.random.normal(0, 5000)
        cost = (areas[i] * rate) + (rooms[i] * 2000) + noise
        costs.append(round(cost, 2))
        
    df = pd.DataFrame({
        "TotalArea": areas,
        "RoomCount": rooms,
        "LuxuryTier": tiers,
        "EstimatedCost": costs
    })
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df

def train_and_evaluate():
    # 1. Load Data
    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        df = generate_data()

    X = df[["TotalArea", "RoomCount", "LuxuryTier"]] 
    y = df["EstimatedCost"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Define Models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    results = []
    best_model = None
    best_mae = float("inf")

    print("\nTraining and Evaluating Models...")
    print("-" * 50)
    print(f"{'Model Name':<20} | {'MAE ($)':<12} | {'R2 Score':<10}")
    print("-" * 50)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({"Model": name, "MAE": mae})
        
        print(f"{name:<20} | ${mae:,.2f}    | {r2:.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_model = model

    print("-" * 50)
    print(f"Winner: {type(best_model).__name__} with MAE: ${best_mae:,.2f}")

    # 3. Plot Comparison
    model_names = [r['Model'] for r in results]
    maes = [r['MAE'] for r in results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, maes, color=['skyblue', 'lightgreen', 'orange', 'salmon'])
    plt.title('Algorithm Performance Comparison (Lower MAE is Better)')
    plt.ylabel('Mean Absolute Error ($)')
    plt.xlabel('Algorithm')
    
    # Add labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"${yval:,.0f}", ha='center', va='bottom')
        
    plt.savefig(CHART_PATH)
    print(f"\nAnalysis Chart Saved: {CHART_PATH}")

    # 4. Save Best Model
    joblib.dump(best_model, MODEL_PATH)
    print(f"Global Best Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
