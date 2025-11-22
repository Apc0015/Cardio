"""
Model Utilities - CardioFusion Clinical Platform
Handles model loading, prediction, and ensemble operations
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import warnings
import hashlib
import os
warnings.filterwarnings('ignore')

# Try to import TensorFlow/Keras for neural network models
try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False


class KerasModelWrapper:
    """
    Wrapper to make Keras models compatible with scikit-learn interface
    Provides predict() and predict_proba() methods
    """
    
    def __init__(self, keras_model):
        """
        Initialize wrapper with a Keras model
        
        Args:
            keras_model: Loaded Keras Sequential or Functional model
        """
        self.model = keras_model
        self.n_features_in_ = keras_model.input_shape[1]
    
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Input features
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        # Get predictions from Keras model (assumes binary classification with sigmoid output)
        predictions = self.model.predict(X, verbose=0)
        
        # Keras model outputs probability of class 1
        # Convert to (n_samples, 2) format: [prob_class_0, prob_class_1]
        prob_class_1 = predictions.flatten()
        prob_class_0 = 1 - prob_class_1
        
        return np.column_stack([prob_class_0, prob_class_1])



class ModelPredictor:
    """
    Professional-grade model prediction handler
    Manages multiple models and ensemble predictions
    """

    def __init__(self, models_dir: str = 'models'):
        """
        Initialize predictor with model directory

        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.feature_names = None
        self.scaler = None
        self.label_encoder = None

    def _verify_file_integrity(self, file_path: Path) -> bool:
        """
        Verify model file integrity before loading
        
        Args:
            file_path: Path to model file
            
        Returns:
            bool: True if file is valid
        """
        try:
            # Check if file exists and is readable
            if not file_path.exists() or not file_path.is_file():
                print(f"❌ File not found: {file_path}")
                return False
                
            # Check file size (should not be empty)
            if file_path.stat().st_size == 0:
                print(f"❌ Empty file: {file_path}")
                return False
                
            # Try to read file header to verify it's a valid pickle file
            with open(file_path, 'rb') as f:
                # Read first few bytes to verify pickle format
                header = f.read(10)
                if not header.startswith(b'\x80\x04'):  # pickle magic number
                    print(f"❌ Invalid pickle format: {file_path}")
                    return False
                    
            return True
            
        except Exception as e:
            print(f"❌ Error verifying {file_path}: {e}")
            return False

    def _validate_model_object(self, model: Any, model_name: str) -> bool:
        """
        Validate loaded model object has required methods
        
        Args:
            model: Loaded model object
            model_name: Name of model
            
        Returns:
            bool: True if model is valid
        """
        try:
            # Check if model has required methods
            required_methods = ['predict', 'predict_proba']
            for method in required_methods:
                if not hasattr(model, method):
                    print(f"❌ Model {model_name} missing method: {method}")
                    return False
                    
            # Test model with dummy data
            if hasattr(model, 'n_features_in_'):
                # Create dummy data with correct number of features
                n_features = model.n_features_in_
                dummy_data = np.random.random((1, n_features))
                
                # Test predict method
                try:
                    model.predict(dummy_data)
                except Exception as e:
                    print(f"❌ Model {model_name} predict method failed: {e}")
                    return False
                    
                # Test predict_proba method
                try:
                    proba = model.predict_proba(dummy_data)
                    if len(proba[0]) != 2:  # Binary classification should have 2 classes
                        print(f"❌ Model {model_name} should output 2 classes, got {len(proba[0])}")
                        return False
                except Exception as e:
                    print(f"❌ Model {model_name} predict_proba method failed: {e}")
                    return False
                    
            return True
            
        except Exception as e:
            print(f"❌ Error validating model {model_name}: {e}")
            return False

    def load_models(self) -> bool:
        """
        Load all available trained models with integrity checks

        Returns:
            bool: Success status
        """
        try:
            # Load preprocessing components
            self._load_preprocessing_components()

            # Load baseline models
            self._load_baseline_models()

            # Load advanced models (if available)
            self._load_advanced_models()

            print(f"✅ Successfully loaded {len(self.models)} models")
            return True

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

    def _load_preprocessing_components(self):
        """Load scaler and label encoder with integrity checks"""
        try:
            # Load from models/preprocessing/ directory
            preprocessing_dir = self.models_dir / 'preprocessing'
            scaler_path = preprocessing_dir / 'scaler.pkl'
            encoder_path = preprocessing_dir / 'label_encoder.pkl'

            if self._verify_file_integrity(scaler_path):
                try:
                    self.scaler = joblib.load(scaler_path)
                    print("✅ Scaler loaded and validated")
                except Exception as e:
                    print(f"⚠️ Failed to load scaler: {e}")
                    
            if self._verify_file_integrity(encoder_path):
                try:
                    self.label_encoder = joblib.load(encoder_path)
                    print("✅ Label encoder loaded and validated")
                except Exception as e:
                    print(f"⚠️ Failed to load label encoder: {e}")
            
            if self.scaler and self.label_encoder:
                print("✅ All preprocessing components loaded successfully")
            else:
                print("⚠️ Some preprocessing components not found or invalid")
        except Exception as e:
            print(f"⚠️ Error loading preprocessing components: {e}")

    def _load_baseline_models(self):
        """Load baseline machine learning models with integrity checks"""
        baseline_dir = self.models_dir / 'baseline_models'

        model_files = {
            'Logistic Regression': 'logistic_regression_model.pkl',
            'Decision Tree': 'decision_tree_model.pkl',
            'Random Forest': 'random_forest_model.pkl'
        }

        for name, filename in model_files.items():
            model_path = baseline_dir / filename
            
            if self._verify_file_integrity(model_path):
                try:
                    model = joblib.load(model_path)
                    
                    # Validate model object
                    if self._validate_model_object(model, name):
                        self.models[name] = model
                        print(f"  📊 Loaded and validated: {name}")
                    else:
                        print(f"  ❌ Model validation failed: {name}")
                        
                except Exception as e:
                    print(f"  ❌ Failed to load {name}: {e}")
            else:
                print(f"  ⚠️ Skipping {name} - file integrity check failed")

    def _load_advanced_models(self):
        """Load advanced models (XGBoost, Neural Network, Ensemble) with integrity checks"""
        advanced_dir = self.models_dir / 'advanced_models'

        if not advanced_dir.exists():
            print("ℹ️ Advanced models not yet trained")
            return

        # Regular models to load as pickle files
        model_files = {
            'XGBoost': 'xgboost_model.pkl',
            'Hybrid Ensemble': 'hybrid_ensemble_model.pkl'
        }

        for name, filename in model_files.items():
            model_path = advanced_dir / filename
            
            if self._verify_file_integrity(model_path):
                try:
                    model = joblib.load(model_path)
                    
                    # Validate model object
                    if self._validate_model_object(model, name):
                        self.models[name] = model
                        print(f"  🚀 Loaded and validated: {name}")
                    else:
                        print(f"  ❌ Model validation failed: {name}")
                        
                except Exception as e:
                    print(f"  ❌ Failed to load {name}: {e}")
            else:
                print(f"  ⚠️ Skipping {name} - file integrity check failed")

        # Load Neural Network model (supports both .pkl and .h5 formats)
        self._load_neural_network(advanced_dir)

    def _load_neural_network(self, advanced_dir: Path):
        """
        Load Neural Network model with support for both .pkl and .h5 formats
        
        Args:
            advanced_dir: Directory containing advanced models
        """
        name = 'Neural Network'
        
        # Try loading .pkl format first
        pkl_path = advanced_dir / 'neural_network_model.pkl'
        if pkl_path.exists() and self._verify_file_integrity(pkl_path):
            try:
                model = joblib.load(pkl_path)
                if self._validate_model_object(model, name):
                    self.models[name] = model
                    print(f"  🧠 Loaded and validated: {name} (pickle format)")
                    return
            except Exception as e:
                print(f"  ⚠️ Failed to load {name} from .pkl: {e}")
        
        # Try loading .h5 format (Keras model)
        h5_path = advanced_dir / 'neural_network_model.h5'
        if h5_path.exists():
            if not KERAS_AVAILABLE:
                print(f"  ⚠️ Skipping {name} - TensorFlow/Keras not available for .h5 models")
                return
                
            try:
                # Load Keras model
                keras_model = keras.models.load_model(h5_path)
                
                # Wrap Keras model to make it compatible with scikit-learn interface
                wrapped_model = KerasModelWrapper(keras_model)
                
                # Validate wrapped model
                if self._validate_model_object(wrapped_model, name):
                    self.models[name] = wrapped_model
                    print(f"  🧠 Loaded and validated: {name} (Keras .h5 format)")
                    return
                else:
                    print(f"  ❌ Model validation failed: {name}")
                    
            except Exception as e:
                print(f"  ⚠️ Failed to load {name} from .h5: {e}")
        
        print(f"  ⚠️ {name} not found in either .pkl or .h5 format")

    def predict(self,
                input_data: pd.DataFrame,
                model_name: Optional[str] = None) -> Dict:
        """
        Make prediction on input data

        Args:
            input_data: Patient data as DataFrame
            model_name: Specific model to use (None = ensemble)

        Returns:
            Dictionary with prediction results
        """
        # CRITICAL: Scale only numerical features (models were trained on scaled numerical data)
        if self.scaler is not None and hasattr(self.scaler, 'feature_names_in_'):
            # Get numerical feature names that need scaling
            numerical_features = self.scaler.feature_names_in_.tolist()

            # Separate numerical and categorical features
            numerical_data = input_data[numerical_features]
            other_features = [col for col in input_data.columns if col not in numerical_features]

            # Scale numerical features
            scaled_numerical = pd.DataFrame(
                self.scaler.transform(numerical_data),
                columns=numerical_features,
                index=input_data.index
            )

            # Combine scaled numerical with unscaled categorical features
            if other_features:
                scaled_data = pd.concat([scaled_numerical, input_data[other_features]], axis=1)
                # Ensure column order matches original
                scaled_data = scaled_data[input_data.columns]
            else:
                scaled_data = scaled_numerical
        else:
            scaled_data = input_data
            print("⚠️ Warning: No scaler loaded or invalid scaler, using raw data")

        if model_name and model_name in self.models:
            return self._single_model_prediction(scaled_data, model_name)
        else:
            return self._ensemble_prediction(scaled_data)

    def _single_model_prediction(self,
                                   data: pd.DataFrame,
                                   model_name: str) -> Dict:
        """
        Prediction from a single model

        Args:
            data: Input features
            model_name: Name of model to use

        Returns:
            Prediction results dictionary
        """
        model = self.models[model_name]

        # Get probability predictions
        proba = model.predict_proba(data)[0]
        prediction = model.predict(data)[0]

        # Get prediction label
        if self.label_encoder is not None:
            prediction_label = self.label_encoder.inverse_transform([prediction])[0]
        else:
            prediction_label = "Disease" if prediction == 1 else "No Disease"

        return {
            'model': model_name,
            'prediction': int(prediction),
            'prediction_label': prediction_label,
            'probability_no_disease': float(proba[0]),
            'probability_disease': float(proba[1]),
            'confidence': float(max(proba)),
            'risk_percentage': float(proba[1] * 100)
        }

    def _ensemble_prediction(self, data: pd.DataFrame) -> Dict:
        """
        Ensemble prediction using all available models

        Args:
            data: Input features

        Returns:
            Aggregated prediction results
        """
        predictions = []
        probabilities = []

        # Define model weights (higher for better performers)
        weights = {
            'Decision Tree': 0.30,
            'Random Forest': 0.25,
            'Logistic Regression': 0.15,
            'XGBoost': 0.35,  # If available
            'Neural Network': 0.10,  # If available
            'Hybrid Ensemble': 0.50  # Highest weight if available
        }

        # Collect predictions from all models
        model_results = {}
        total_weight = 0

        for name, model in self.models.items():
            weight = weights.get(name, 0.15)
            proba = model.predict_proba(data)[0]

            model_results[name] = {
                'probability_disease': float(proba[1]),
                'weight': weight
            }

            probabilities.append(proba[1] * weight)
            total_weight += weight

        # Calculate weighted average
        ensemble_prob = sum(probabilities) / total_weight if total_weight > 0 else 0.5
        ensemble_prediction = 1 if ensemble_prob >= 0.5 else 0

        # Get prediction label
        if self.label_encoder is not None:
            prediction_label = self.label_encoder.inverse_transform([ensemble_prediction])[0]
        else:
            prediction_label = "Disease" if ensemble_prediction == 1 else "No Disease"

        return {
            'model': 'Ensemble (Weighted Average)',
            'prediction': int(ensemble_prediction),
            'prediction_label': prediction_label,
            'probability_no_disease': float(1 - ensemble_prob),
            'probability_disease': float(ensemble_prob),
            'confidence': float(max(ensemble_prob, 1 - ensemble_prob)),
            'risk_percentage': float(ensemble_prob * 100),
            'individual_models': model_results
        }

    def get_risk_category(self, risk_percentage: float) -> Tuple[str, str, str]:
        """
        Categorize risk level with clinical interpretation

        Args:
            risk_percentage: Risk score (0-100)

        Returns:
            Tuple of (category, emoji, color)
        """
        if risk_percentage < 30:
            return ("Low Risk", "✅", "#059669")
        elif risk_percentage < 50:
            return ("Moderate-Low Risk", "⚠️", "#f59e0b")
        elif risk_percentage < 70:
            return ("Moderate-High Risk", "⚠️", "#d97706")
        else:
            return ("High Risk", "🚨", "#dc2626")

    def get_available_models(self) -> List[str]:
        """Get list of loaded model names"""
        return list(self.models.keys())

    def get_model_health_status(self) -> Dict[str, Dict]:
        """
        Get health status of all loaded models
        
        Returns:
            Dictionary with model health information
        """
        health_status = {}
        
        for name, model in self.models.items():
            try:
                # Test model with dummy data
                if hasattr(model, 'n_features_in_'):
                    n_features = model.n_features_in_
                    dummy_data = np.random.random((1, n_features))
                    
                    # Quick prediction test
                    pred = model.predict(dummy_data)
                    proba = model.predict_proba(dummy_data)
                    
                    health_status[name] = {
                        'status': 'healthy',
                        'features': n_features,
                        'test_prediction': 'passed',
                        'test_probability': 'passed'
                    }
                else:
                    health_status[name] = {
                        'status': 'unknown',
                        'reason': 'Cannot determine feature count'
                    }
                    
            except Exception as e:
                health_status[name] = {
                    'status': 'error',
                    'error': str(e)
                }
                
        return health_status


def load_all_models(models_dir: str = 'models') -> ModelPredictor:
    """
    Convenience function to create and load ModelPredictor

    Args:
        models_dir: Directory containing models

    Returns:
        Initialized ModelPredictor
    """
    predictor = ModelPredictor(models_dir)
    predictor.load_models()
    return predictor


def format_prediction_output(prediction: Dict, detailed: bool = False) -> str:
    """
    Format prediction results for display

    Args:
        prediction: Prediction dictionary
        detailed: Show detailed output

    Returns:
        Formatted string output
    """
    risk_pct = prediction['risk_percentage']
    category, emoji, _ = ModelPredictor(None).get_risk_category(risk_pct)

    if not detailed:
        # Simple output
        return f"""
{emoji} **{category}**
Risk Score: {risk_pct:.1f}%
Prediction: {prediction['prediction_label']}
        """
    else:
        # Detailed output
        output = f"""
╔═════════════════════════════════╗
║  🩺 CARDIOVASCULAR RISK ASSESSMENT    ║
╠═══════════════════════════════════╣

📊 RISK LEVEL: {emoji} {category.upper()}
├─ No Disease Probability: {prediction['probability_no_disease']*100:.1f}%
└─ Heart Disease Probability: {prediction['probability_disease']*100:.1f}%

🎯 PREDICTION: {prediction['prediction_label']}
📈 Confidence: {prediction['confidence']*100:.1f}%
🤖 Model: {prediction['model']}
        """

        if 'individual_models' in prediction:
            output += "\n\n📊 INDIVIDUAL MODEL PREDICTIONS:\n"
            for model, results in prediction['individual_models'].items():
                output += f"├─ {model}: {results['probability_disease']*100:.1f}% (weight: {results['weight']:.2f})\n"

        output += "\n╚═════════════════════════════════╝"
        return output
