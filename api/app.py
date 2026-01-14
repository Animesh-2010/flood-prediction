from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.predictor import FloodPredictor

app = Flask(__name__)
CORS(app)

predictor = None

@app.before_request
def load_model():
    global predictor
    if predictor is None:
        predictor = FloodPredictor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Flood Prediction API v2.0'
    }), 200

@app.route('/api/v1/predict/risk', methods=['POST'])
def predict_flood_risk():
    """Predict flood risk (classification)."""
    try:
        data = request.get_json()
        
        required_fields = [
            'rainfall_mm_per_hr', 'river_level_m', 'soil_moisture_pct',
            'temperature_c', 'humidity_pct', 'wind_speed_kmh', 'elevation_m',
            'distance_to_river_m', 'previous_day_rainfall_mm', 'catchment_area_rainfall_mm'
        ]
        
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                'error': 'Missing fields',
                'missing_fields': missing
            }), 400
        
        result = predictor.predict_flood_risk(data)
        
        return jsonify({
            'success': True,
            'prediction': result
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/v1/predict/water-level', methods=['POST'])
def predict_water_level():
    """Predict water level 24h ahead."""
    try:
        data = request.get_json()
        result = predictor.predict_water_level_24h(data)
        
        return jsonify({
            'success': True,
            'prediction': result
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/v1/predict/all', methods=['POST'])
def predict_all():
    """Get both flood risk and water level predictions."""
    try:
        data = request.get_json()
        result = predictor.predict_all(data)
        
        return jsonify({
            'success': True,
            'prediction': result
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/v1/models/info', methods=['GET'])
def model_info():
    """Get info about loaded models."""
    return jsonify({
        'models': {
            'classification': ['Random Forest', 'XGBoost (optimized)'],
            'regression': ['LSTM (optimized)', 'GRU (optimized)'],
            'ensemble': 'Average probability/output ensemble'
        },
        'features': predictor.feature_cols,
        'risk_levels': predictor.risk_levels,
        'improvements': [
            'Hyperparameter tuning (RandomizedSearchCV)',
            'Batch normalization in deep models',
            'Native Keras (.keras) format',
            'Better early stopping',
            'Adam optimizer with reduced learning rate'
        ]
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
