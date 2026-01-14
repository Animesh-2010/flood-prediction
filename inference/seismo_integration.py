from predictor import FloodPredictor
import pandas as pd
from datetime import datetime

class SeismoFloodModule:
    """Bridge between Seismo data pipeline and flood prediction."""
    
    def __init__(self):
        self.predictor = FloodPredictor()
        
    def process_sensor_data(self, sensor_payload):
        """
        Process incoming sensor data from Seismo IoT devices.
        
        Expected sensor_payload format:
        {
            'device_id': 'seismo_001',
            'timestamp': '2024-01-15T10:30:00Z',
            'sensors': {
                'rainfall': 15.5,      # mm/hr
                'water_level': 4.2,    # m
                'soil_moisture': 75.0, # %
                'temperature': 28.0,   # °C
                'humidity': 80.0,      # %
                'wind_speed': 12.0,    # km/h
                'elevation': 50.0,     # m (relative)
                'distance_to_river': 500.0  # m
            },
            'metadata': {
                'latitude': 13.0827,
                'longitude': 77.6064,
                'location_name': 'Bengaluru, Karnataka'
            }
        }
        """
        
        try:
            # Extract sensor data
            sensors = sensor_payload['sensors']
            
            # Build prediction input
            prediction_input = {
                'rainfall_mm_per_hr': sensors.get('rainfall', 0),
                'river_level_m': sensors.get('water_level', 0),
                'soil_moisture_pct': sensors.get('soil_moisture', 50),
                'temperature_c': sensors.get('temperature', 25),
                'humidity_pct': sensors.get('humidity', 60),
                'wind_speed_kmh': sensors.get('wind_speed', 0),
                'elevation_m': sensors.get('elevation', 0),
                'distance_to_river_m': sensors.get('distance_to_river', 1000),
                'previous_day_rainfall_mm': sensor_payload.get('cumulative_rainfall_24h', 0),
                'catchment_area_rainfall_mm': sensors.get('rainfall', 0) * 0.8
            }
            
            # Get predictions
            result = self.predictor.predict_all(prediction_input)
            
            # Format for Seismo output
            output = {
                'device_id': sensor_payload['device_id'],
                'timestamp': datetime.now().isoformat(),
                'location': sensor_payload['metadata'],
                'predictions': {
                    'flood_risk': result['flood_risk'],
                    'water_level_24h': result['water_level_24h']
                },
                'alert_triggered': result['flood_risk']['risk_level'] in ['HIGH', 'CRITICAL']
            }
            
            return output
            
        except Exception as e:
            return {
                'error': str(e),
                'device_id': sensor_payload.get('device_id'),
                'timestamp': datetime.now().isoformat()
            }

# Example: Integrate into Seismo's real-time handler
def handle_seismo_mqtt(client, userdata, msg):
    """MQTT callback function for Seismo."""
    import json
    
    payload = json.loads(msg.payload.decode())
    
    seismo_flood = SeismoFloodModule()
    prediction = seismo_flood.process_sensor_data(payload)
    
    # Publish alert if needed
    if prediction.get('alert_triggered'):
        client.publish(
            f"seismo/alerts/{payload['device_id']}",
            json.dumps(prediction)
        )
    
    # Log prediction
    print(f"Prediction: {prediction}")