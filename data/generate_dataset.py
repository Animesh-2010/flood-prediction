import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_flood_dataset(n_samples=5000, random_state=42):
    """
    Generate realistic synthetic flood prediction dataset.
    
    Features:
    - rainfall (mm/hour): 0-50
    - river_level (m): 0-10
    - soil_moisture (%): 0-100
    - temperature (°C): 5-40
    - humidity (%): 30-95
    - wind_speed (km/h): 0-30
    - elevation (m): 0-500 (relative to river)
    - distance_to_river (m): 0-5000
    - previous_day_rainfall (mm): 0-200
    - catchment_area_rain (mm/hour): 0-40
    
    Target:
    - flood_risk: 0 (no flood), 1 (low), 2 (medium), 3 (high)
    - water_level_24h (m): predicted water level 24h ahead
    """
    
    np.random.seed(random_state)
    
    # Timestamp
    start_date = datetime(2018, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Base features (independent)
    rainfall = np.random.exponential(scale=2.0, size=n_samples)
    rainfall[rainfall > 50] = 50  # Clip to 50mm/hr max
    
    river_level = np.random.uniform(0.5, 5.0, n_samples)
    soil_moisture = np.random.uniform(20, 95, n_samples)
    temperature = np.random.uniform(5, 40, n_samples)
    humidity = np.random.uniform(30, 95, n_samples)
    wind_speed = np.random.uniform(0, 25, n_samples)
    elevation = np.random.uniform(0, 500, n_samples)
    distance_to_river = np.random.uniform(100, 5000, n_samples)
    
    # Lagged features (time-dependent)
    previous_day_rainfall = np.zeros(n_samples)
    catchment_area_rain = np.zeros(n_samples)
    
    for i in range(1, n_samples):
        # Moving average of past 24 hours (simplified)
        look_back = min(24, i)
        previous_day_rainfall[i] = np.mean(rainfall[max(0, i-24):i])
        catchment_area_rain[i] = rainfall[i] * 0.8 + np.random.normal(0, 1)
    
    # Target: flood risk classification based on combined factors
    # Risk increases with: high rainfall, high river level, high soil moisture
    risk_score = (
        rainfall * 0.4 +                           # Current rainfall weight
        river_level * 0.25 +                       # River level weight
        (soil_moisture / 100) * 0.2 +              # Soil saturation weight
        (previous_day_rainfall / 200) * 0.15      # Cumulative rainfall weight
    )
    
    # Add noise
    risk_score += np.random.normal(0, 0.1, n_samples)
    risk_score = np.clip(risk_score, 0, 1)
    
    # Convert to classification (0-3)
    flood_risk = np.digitize(risk_score, bins=[0.25, 0.5, 0.75]) 
    
    # Water level 24h ahead (continuous prediction target)
    water_level_24h = (
        river_level * 0.7 +
        (rainfall / 10) * 0.2 +
        (soil_moisture / 100) * 0.1 +
        np.random.normal(0, 0.3, n_samples)
    )
    water_level_24h = np.clip(water_level_24h, 0, 8)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'rainfall_mm_per_hr': rainfall,
        'river_level_m': river_level,
        'soil_moisture_pct': soil_moisture,
        'temperature_c': temperature,
        'humidity_pct': humidity,
        'wind_speed_kmh': wind_speed,
        'elevation_m': elevation,
        'distance_to_river_m': distance_to_river,
        'previous_day_rainfall_mm': previous_day_rainfall,
        'catchment_area_rainfall_mm': catchment_area_rain,
        'flood_risk_class': flood_risk,           # 0-3 classification
        'water_level_24h_m': water_level_24h      # Continuous target
    })
    
    return df

def main():
    print("Generating synthetic flood dataset...")
    df = generate_synthetic_flood_dataset(n_samples=5000)
    
    # Save
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/flood_data_raw.csv', index=False)
    
    print(f"✓ Dataset generated: {len(df)} samples")
    print(f"\nDataset Summary:")
    print(df.describe())
    print(f"\nFlood Risk Distribution:")
    print(df['flood_risk_class'].value_counts().sort_index())
    print(f"\nMissing Values: {df.isnull().sum().sum()}")

if __name__ == '__main__':
    main()