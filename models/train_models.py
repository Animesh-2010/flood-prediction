import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    auc, roc_curve, mean_squared_error, mean_absolute_error, f1_score
)
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

class FloodModelTrainer:
    def __init__(self, data_path='data/raw/flood_data_raw.csv'):
        self.df = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        self.scaler_minmax = MinMaxScaler()
        self.models = {}
        self.history = {}
        
    def prepare_data(self, test_size=0.2, val_size=0.1):
        """Prepare and scale data for training."""
        
        feature_cols = [
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
        
        X = self.df[feature_cols].values
        y_class = self.df['flood_risk_class'].values
        y_reg = self.df['water_level_24h_m'].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X_scaled, y_class, y_reg, test_size=test_size, random_state=42, stratify=y_class
        )
        
        X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val = train_test_split(
            X_train, y_class_train, y_reg_train, 
            test_size=val_size/(1-test_size), random_state=42, stratify=y_class_train
        )
        
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_class_train, self.y_class_val, self.y_class_test = y_class_train, y_class_val, y_class_test
        self.y_reg_train, self.y_reg_val, self.y_reg_test = y_reg_train, y_reg_val, y_reg_test
        self.feature_cols = feature_cols
        
        print(f"✓ Data prepared:")
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
    def train_random_forest(self):
        """Train optimized Random Forest for flood classification."""
        print("\n[1/4] Training Random Forest (with hyperparameter tuning)...")
        
        rf_base = RandomForestClassifier(n_jobs=-1, random_state=42)
        
        # Hyperparameter tuning space [web:58]
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 4, 8],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False]
        }
        
        random_search = RandomizedSearchCV(
            rf_base, param_dist, n_iter=20, cv=3, 
            scoring='roc_auc_ovr', n_jobs=-1, random_state=42
        )
        
        random_search.fit(self.X_train, self.y_class_train)
        rf = random_search.best_estimator_
        
        print(f"  Best params: {random_search.best_params_}")
        
        # Evaluate
        y_pred = rf.predict(self.X_test)
        y_pred_proba = rf.predict_proba(self.X_test)
        
        auc_score = roc_auc_score(
        self.y_class_test,
        y_pred_proba,
        multi_class='ovr',
        average='weighted'
        )

        
        print(f"  ✓ AUC Score: {auc_score:.4f}")
        print(f"  ✓ Classification Report:")
        print(classification_report(
            self.y_class_test, y_pred, 
            zero_division=0
        ))
        
        self.models['random_forest'] = rf
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Top Features:")
        print(feature_importance.head())
        
        return rf
    
    def train_xgboost(self):
        """Train optimized XGBoost for flood classification."""
        print("\n[2/4] Training XGBoost (with hyperparameter tuning)...")
        
        xgb_base = xgb.XGBClassifier(
            n_jobs=-1, random_state=42, eval_metric='mlogloss',
            tree_method='hist'
        )
        
        # Hyperparameter tuning space [web:52][web:54]
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': stats.uniform(0.01, 0.2),
            'subsample': stats.uniform(0.5, 0.5),
            'colsample_bytree': stats.uniform(0.5, 0.5),
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.5, 1],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1, 5]
        }
        
        random_search = RandomizedSearchCV(
            xgb_base, param_dist, n_iter=30, cv=3,
            scoring='roc_auc_ovr', n_jobs=-1, random_state=42
        )
        
        random_search.fit(
            self.X_train, self.y_class_train,
            eval_set=[(self.X_val, self.y_class_val)],
            verbose=False
        )
        xgb_model = random_search.best_estimator_
        
        print(f"  Best params: {random_search.best_params_}")
        
        # Evaluate
        y_pred = xgb_model.predict(self.X_test)
        y_pred_proba = xgb_model.predict_proba(self.X_test)
        auc_score = roc_auc_score(self.y_class_test, y_pred_proba, multi_class='ovr', average='weighted')
        
        print(f"  ✓ AUC Score: {auc_score:.4f}")
        print(f"  ✓ Classification Report:")
        print(classification_report(
            self.y_class_test, y_pred,
            zero_division=0
        ))
        
        self.models['xgboost'] = xgb_model
        return xgb_model
    
    def train_lstm(self):
        """Train optimized LSTM for water level regression."""
        print("\n[3/4] Training LSTM (optimized architecture)...")
        
        # Reshape for LSTM [web:53]
        X_train_lstm = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
        X_val_lstm = self.X_val.reshape((self.X_val.shape[0], 1, self.X_val.shape[1]))
        X_test_lstm = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
        
        # Build model with batch normalization and better regularization
        model = Sequential([
            keras.Input(shape=(1, self.X_train.shape[1])),
            LSTM(128, activation='relu', return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, activation='relu', return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        # Use explicit loss function object
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss', patience=15, 
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, 
            patience=7, min_lr=1e-7
        )
        
        history = model.fit(
            X_train_lstm, self.y_reg_train,
            validation_data=(X_val_lstm, self.y_reg_val),
            epochs=150,
            batch_size=16,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Evaluate
        y_pred = model.predict(X_test_lstm, verbose=0).flatten()
        rmse = np.sqrt(mean_squared_error(self.y_reg_test, y_pred))
        mae = mean_absolute_error(self.y_reg_test, y_pred)
        r2 = 1 - (np.sum((self.y_reg_test - y_pred)**2) / np.sum((self.y_reg_test - np.mean(self.y_reg_test))**2))
        
        print(f"  ✓ RMSE: {rmse:.4f} m")
        print(f"  ✓ MAE: {mae:.4f} m")
        print(f"  ✓ R²: {r2:.4f}")
        
        # Save in native Keras format
        model.save('models/saved_models/lstm_flood.keras')
        self.models['lstm'] = model
        self.history['lstm'] = history
        
        return model
    
    def train_gru(self):
        """Train optimized GRU for water level regression."""
        print("\n[4/4] Training GRU (optimized architecture)...")
        
        # Reshape for GRU [web:53]
        X_train_gru = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
        X_val_gru = self.X_val.reshape((self.X_val.shape[0], 1, self.X_val.shape[1]))
        X_test_gru = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
        
        # Build model - GRU with 2 layers performs better [web:53]
        model = Sequential([
            keras.Input(shape=(1, self.X_train.shape[1])),
            GRU(128, activation='relu', return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            GRU(64, activation='relu', return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=7, min_lr=1e-7
        )
        
        history = model.fit(
            X_train_gru, self.y_reg_train,
            validation_data=(X_val_gru, self.y_reg_val),
            epochs=150,
            batch_size=16,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Evaluate
        y_pred = model.predict(X_test_gru, verbose=0).flatten()
        rmse = np.sqrt(mean_squared_error(self.y_reg_test, y_pred))
        mae = mean_absolute_error(self.y_reg_test, y_pred)
        r2 = 1 - (np.sum((self.y_reg_test - y_pred)**2) / np.sum((self.y_reg_test - np.mean(self.y_reg_test))**2))
        
        print(f"  ✓ RMSE: {rmse:.4f} m")
        print(f"  ✓ MAE: {mae:.4f} m")
        print(f"  ✓ R²: {r2:.4f}")
        
        # Save in native Keras format
        model.save('models/saved_models/gru_flood.keras')
        self.models['gru'] = model
        self.history['gru'] = history
        
        return model
    
    def save_all_models(self):
        """Save trained models."""
        import os
        os.makedirs('models/saved_models', exist_ok=True)
        
        joblib.dump(self.models['random_forest'], 'models/saved_models/rf_flood.pkl')
        joblib.dump(self.models['xgboost'], 'models/saved_models/xgb_flood.pkl')
        joblib.dump(self.scaler, 'models/saved_models/scaler.pkl')
        
        print("\n✓ All models saved to models/saved_models/")
    
    def train_all(self):
        """Train all models in sequence."""
        self.prepare_data()
        self.train_random_forest()
        self.train_xgboost()
        self.train_lstm()
        self.train_gru()
        self.save_all_models()
        
        print("\n" + "="*60)
        print("✓ All models trained successfully!")
        print("="*60)

if __name__ == '__main__':
    trainer = FloodModelTrainer()
    trainer.train_all()
