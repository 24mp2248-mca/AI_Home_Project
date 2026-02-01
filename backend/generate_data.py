import pandas as pd
import numpy as np
import random

def generate_synthetic_data(num_samples=5000):
    data = []
    
    for _ in range(num_samples):
        # Basic features
        total_area = np.random.normal(150, 60) # Larger mean
        total_area = max(40, total_area) # Min 40m2
        
        num_bedrooms = int(np.clip(np.random.normal(3, 1), 1, 8))
        num_bathrooms = int(np.clip(np.random.normal(2, 1), 1, 6))
        num_living_rooms = np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05])
        num_kitchens = np.random.choice([1, 2], p=[0.9, 0.1])
        
        num_rooms = num_bedrooms + num_bathrooms + num_living_rooms + num_kitchens
        
        # New Features: Quality and Complexity
        # Quality (1-10): 1=Basic, 5=Standard, 10=Luxury
        quality_score = np.random.randint(3, 11) 
        
        # Complexity (1-10): Corner counts, non-rectangular shapes
        complexity_score = np.random.randint(1, 11)
        
        # --- COST CALCULATION ---
        # Base construction cost: $1200 per m2 (Basic) to $3500+ (Luxury)
        base_rate = 1200 + (quality_score * 250)
        
        # Complexity multiplier (1.0 to 1.3)
        complexity_mult = 1.0 + (complexity_score * 0.03)
        
        # Feature cost modifiers
        cost_bedrooms = num_bedrooms * 5000 * (quality_score / 5)
        cost_bathrooms = num_bathrooms * 12000 * (quality_score / 5) # Baths are expensive
        cost_kitchens = num_kitchens * 25000 * (quality_score / 5)   # Kitchens are very expensive
        cost_living = num_living_rooms * 8000 * (quality_score / 5)
        
        # Random noise
        noise = np.random.normal(0, 8000)
        
        total_cost = (total_area * base_rate * complexity_mult) + \
                     cost_bedrooms + cost_bathrooms + cost_kitchens + cost_living + noise
        
        data.append({
            'total_area': round(total_area, 2),
            'num_rooms': num_rooms,
            'num_bedrooms': num_bedrooms,
            'num_bathrooms': num_bathrooms,
            'num_living_rooms': num_living_rooms,
            'num_kitchens': num_kitchens,
            'quality_score': quality_score,
            'complexity_score': complexity_score,
            'estimated_cost': round(total_cost, 2)
        })
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_synthetic_data()
    # Save to CSV
    df.to_csv("housing_data.csv", index=False)
    print(f"Generated housing_data.csv with {len(df)} samples.")
