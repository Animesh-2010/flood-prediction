import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FloodPredictor:
    """Production inference engine for flood prediction models."""
    
    def __init__(self, model_dir='models/saved_models'):
        """Load all trained models."""
        self.model_dir = Path(model_dir)
        
        # Load ensemble models
        self.rf_model = joblib.load(self.model_dir / 'rf_flood.pkl')
        self.xgb_model = joblib.load(self.model_dir / 'xgb_flood.pkl')
        self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
        
        # Load deep learning models WITHOUT compiling (no metric deserialization issues)
        self.lstm_model = tf.keras.models.load_model(
            self.model_dir / 'lstm_flood.keras',
            compile=False
        )
        self.gru_model = tf.keras.models.load_model(
            self.model_dir / 'gru_flood.keras',
            compile=False
        )
        
        self.feature_cols = [
            'rainfall_mm_per_hr',
            'river_level_m',
            'soil_moisture_pct',
            'temperature_c',
            'humidity_pct',
            'wind_speed_kmh',
            'elevation_m',
            'distance_to_river_m',
            'previous_day_rainfall_mm',
            'catchment_area_rainfall_mm'
        ]
        
        self.risk_levels = {
            0: {'name': 'LOW', 'color': 'green', 'description': 'No flood threat'},
            1: {'name': 'MEDIUM', 'color': 'yellow', 'description': 'Monitor conditions'},
            2: {'name': 'HIGH', 'color': 'orange', 'description': 'Prepare to evacuate'},
            3: {'name': 'CRITICAL', 'color': 'red', 'description': 'Evacuate immediately'}
        }
        
        print("✓ All models loaded successfully!")
        
    def predict_flood_risk(self, input_dict):
        """
        Predict flood risk (classification).
        
        Args:
            input_dict: Dict with feature columns
            
        Returns:
            Dict with ensemble flood risk prediction
        """
        
        X = np.array([input_dict[col] for col in self.feature_cols]).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        rf_pred = self.rf_model.predict(X_scaled)[0]
        xgb_pred = self.xgb_model.predict(X_scaled)[0]
        
        rf_proba = self.rf_model.predict_proba(X_scaled)[0]
        xgb_proba = self.xgb_model.predict_proba(X_scaled)[0]
        
        ensemble_proba = (rf_proba + xgb_proba) / 2
        ensemble_pred = np.argmax(ensemble_proba)
        confidence = float(np.max(ensemble_proba))
        
        return {
            'rf_prediction': int(rf_pred),
            'xgb_prediction': int(xgb_pred),
            'ensemble_prediction': int(ensemble_pred),
            'risk_level': self.risk_levels[ensemble_pred]['name'],
            'risk_description': self.risk_levels[ensemble_pred]['description'],
            'confidence': confidence,
            'probability_distribution': {
                int(i): float(p) for i, p in enumerate(ensemble_proba)
            }
        }
    
    def predict_water_level_24h(self, input_dict):
        """
        Predict water level 24 hours ahead (regression).
        
        Args:
            input_dict: Dict with feature columns
            
        Returns:
            Dict with water level predictions
        """
        
        X = np.array([input_dict[col] for col in self.feature_cols]).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        X_reshaped = X_scaled.reshape((1, 1, X_scaled.shape[1]))
        
        lstm_pred = float(self.lstm_model.predict(X_reshaped, verbose=0)[0][0])
        gru_pred = float(self.gru_model.predict(X_reshaped, verbose=0)[0][0])
        
        ensemble_pred = (lstm_pred + gru_pred) / 2
        
        return {
            'lstm_prediction_m': lstm_pred,
            'gru_prediction_m': gru_pred,
            'ensemble_prediction_m': ensemble_pred,
            'unit': 'meters'
        }
    
    def predict_all(self, input_dict):
        """
        Get both classification and regression predictions.
        
        Returns:
            Combined result dict
        """
        
        flood_risk = self.predict_flood_risk(input_dict)
        water_level = self.predict_water_level_24h(input_dict)
        
        return {
            'flood_risk': flood_risk,
            'water_level_24h': water_level,
            'timestamp': pd.Timestamp.now().isoformat()
        }

if __name__ == '__main__':
    predictor = FloodPredictor()
    
    sample_input = {
        'rainfall_mm_per_hr': 15.5,
        'river_level_m': 4.2,
        'soil_moisture_pct': 75.0,
        'temperature_c': 28.0,
        'humidity_pct': 80.0,
        'wind_speed_kmh': 12.0,
        'elevation_m': 50.0,
        'distance_to_river_m': 500.0,
        'previous_day_rainfall_mm': 120.0,
        'catchment_area_rainfall_mm': 18.0
    }
    
    result = predictor.predict_all(sample_input)
    
    print("\n" + "="*60)
    print("FLOOD RISK PREDICTION")
    print("="*60)
    print(f"Risk Level: {result['flood_risk']['risk_level']}")
    print(f"Description: {result['flood_risk']['risk_description']}")
    print(f"Confidence: {result['flood_risk']['confidence']:.2%}")
    print(f"Probability Distribution: {result['flood_risk']['probability_distribution']}")
    
    print("\n" + "="*60)
    print("WATER LEVEL PREDICTION (24h ahead)")
    print("="*60)
    print(f"LSTM Prediction: {result['water_level_24h']['lstm_prediction_m']:.3f} m")
    print(f"GRU Prediction: {result['water_level_24h']['gru_prediction_m']:.3f} m")
    print(f"Ensemble Prediction: {result['water_level_24h']['ensemble_prediction_m']:.3f} m")
