import pandas as pd
import numpy as np
import random

def generate_synthetic_data(num_samples=5000):
    data = []
    
    for _ in range(num_samples):
        # 1. Generate Room Counts first
        # Small houses are more common than mansions
        num_bedrooms = int(np.clip(np.random.normal(3, 1), 1, 6))
        num_bathrooms = int(np.clip(np.random.normal(2, 1), 1, 4))
        
        # Logic: Usually 1 living, sometimes 2
        num_living_rooms = np.random.choice([1, 2], p=[0.8, 0.2])
        
        # Logic: 1 Kitchen
        num_kitchens = 1
        
        num_rooms = num_bedrooms + num_bathrooms + num_living_rooms + num_kitchens
        
        # 2. Derive Area from Room Count (Modern Standards)
        # Avg room size 14-24m2 (Spacious) + 25% circulation/walls
        avg_room_size = np.random.uniform(14, 24)
        total_area = (num_rooms * avg_room_size) * 1.25
        total_area = round(total_area, 2)

        # 3. Features
        # Quality (1-10)
        quality_score = np.random.randint(3, 10) 
        # Complexity (1-10)
        complexity_score = np.random.randint(1, 8)
        
        # --- COST CALCULATION (TARGET: $70k - $90k for 100m2) ---
        # Target Cost/m2 (All-in): ~$700 - $900
        # Base Construction Rate: $350 (Basic) to $650 (Luxury) per m2
        base_rate = 350 + (quality_score * 50)
        
        # Complexity multiplier
        complexity_mult = 1.0 + (complexity_score * 0.02)
        
        # Feature cost modifiers (Aligned with local turnkey rates)
        # Bedrooms: ~1.5k
        cost_bedrooms = num_bedrooms * 1500 * (quality_score / 5)
        # Baths: ~4k
        cost_bathrooms = num_bathrooms * 4000 * (quality_score / 5)
        # Kitchens: ~8k (Modular)
        cost_kitchens = num_kitchens * 8000 * (quality_score / 5)
        # Living: ~2k
        cost_living = num_living_rooms * 2000 * (quality_score / 5)
        
        # Random noise
        noise = np.random.normal(0, 1500)
        
        total_cost = (total_area * base_rate * complexity_mult) + \
                     cost_bedrooms + cost_bathrooms + cost_kitchens + cost_living + noise
        
        data.append({
            'total_area': total_area,
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
