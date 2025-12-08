import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuralGaitClassifier(nn.Module):
    """Neural network for gait classification"""
    
    def __init__(self, input_size: int = 20, hidden_size: int = 64, num_classes: int = 5):
        super(NeuralGaitClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)

class GaitClassifier:
    """
    Advanced ML-based gait classifier for Phase 2
    Supports multiple models: Random Forest, Gradient Boosting, Neural Network
    """
    
    def __init__(self, model_dir: Path = Path("ml_models/saved_models")):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Classification labels
        self.class_labels = [
            'neutral',
            'mild_pronation', 
            'moderate_pronation',
            'severe_pronation',
            'supination'
        ]
        
        # Clinical thresholds based on research literature
        # Note: Negative angles = pronation (inward roll), Positive angles = supination (outward roll)
        self.clinical_thresholds = {
            'pronation': {
                'neutral': (-5, 5),
                'mild_pronation': (-10, -5),       # Fixed: negative for inward roll
                'moderate_pronation': (-15, -10),   # Fixed: negative for inward roll
                'severe_pronation': (float('-inf'), -15),  # Fixed: very negative for severe inward roll
                'supination': (5, float('inf'))    # Fixed: positive for outward roll
            },
            'cadence': {
                'slow': (0, 95),
                'normal': (95, 125),
                'fast': (125, float('inf'))
            },
            'step_length': {
                'short': (0, 0.5),
                'normal': (0.5, 0.8),
                'long': (0.8, float('inf'))
            }
        }
        
        # Clinical reference values (from gait analysis research)
        self.reference_values = {
            'normal_cadence': {'mean': 112, 'std': 5.4},
            'normal_step_length': {'mean': 0.67, 'std': 0.05},
            'normal_stance_phase': {'mean': 62, 'std': 2},
            'normal_swing_phase': {'mean': 38, 'std': 2},
            'normal_pronation': {'mean': 0, 'std': 5}
        }
        
        # Initialize models
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.neural_model = NeuralGaitClassifier()
        self.scaler = StandardScaler()
        
        # Load pre-trained models if available
        self._load_models()
        
        # Performance metrics
        self.performance_metrics = {}
        
        # Load pretrained clinical classifier
        try:
            from .pretrained_loader import ClinicalGaitClassifier
            self.clinical_classifier = ClinicalGaitClassifier()
            logger.info("Clinical classifier loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load clinical classifier: {e}")
            self.clinical_classifier = None
        
    def classify_gait(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify gait using ensemble of models with clinical validation
        """
        try:
            feature_vector = np.array(features['feature_vector']).reshape(1, -1)
            
            # Normalize features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Get predictions from all models
            predictions = {}
            confidences = {}
            
            # Random Forest
            if hasattr(self.rf_model, 'predict_proba'):
                rf_proba = self.rf_model.predict_proba(feature_vector_scaled)[0]
                rf_pred = self.rf_model.predict(feature_vector_scaled)[0]
                predictions['random_forest'] = self.class_labels[rf_pred]
                confidences['random_forest'] = float(max(rf_proba))
            
            # Gradient Boosting
            if hasattr(self.gb_model, 'predict_proba'):
                gb_proba = self.gb_model.predict_proba(feature_vector_scaled)[0]
                gb_pred = self.gb_model.predict(feature_vector_scaled)[0]
                predictions['gradient_boosting'] = self.class_labels[gb_pred]
                confidences['gradient_boosting'] = float(max(gb_proba))
            
            # Neural Network
            if self.neural_model and hasattr(self.neural_model, 'forward'):
                with torch.no_grad():
                    nn_input = torch.FloatTensor(feature_vector_scaled)
                    nn_output = self.neural_model(nn_input)
                    nn_proba = nn_output.numpy()[0]
                    nn_pred = np.argmax(nn_proba)
                    predictions['neural_network'] = self.class_labels[nn_pred]
                    confidences['neural_network'] = float(max(nn_proba))
            
            # Ensemble prediction (majority voting with confidence weighting)
            ensemble_prediction = self._ensemble_predict(predictions, confidences)
            
            # Clinical rule-based classification for validation
            clinical_prediction = self._clinical_classification(features)
            
            # FORCE clinical classification until ML models are retrained with correct thresholds
            # The ML models were trained with incorrect angle interpretations
            if clinical_prediction['confidence'] > 0.5:
                logger.info(f"Using clinical classification: {clinical_prediction['class']} (confidence: {clinical_prediction['confidence']:.3f})")
                ensemble_prediction['class'] = clinical_prediction['class']
                ensemble_prediction['confidence'] = clinical_prediction['confidence']
                ensemble_prediction['method_used'] = 'clinical_override'
            elif ensemble_prediction.get('confidence', 0) < 0.6:
                # Blend ML and clinical predictions
                if clinical_prediction['confidence'] > 0.3:
                    # Use clinical prediction with higher weight
                    ensemble_prediction['class'] = clinical_prediction['class']
                    ensemble_prediction['confidence'] = (
                        ensemble_prediction.get('confidence', 0) * 0.3 + 
                        clinical_prediction['confidence'] * 0.7
                    )
                    ensemble_prediction['method_used'] = 'clinical_weighted'
                else:
                    # Average both predictions
                    ensemble_prediction['confidence'] = (
                        ensemble_prediction.get('confidence', 0) * 0.5 + 
                        clinical_prediction['confidence'] * 0.5
                    )
                    ensemble_prediction['method_used'] = 'hybrid'
            else:
                ensemble_prediction['method_used'] = 'ml_ensemble'
            
            # Add clinical validation score
            clinical_validation = self._validate_with_clinical_thresholds(
                ensemble_prediction['class'], 
                features
            )
            
            return {
                'ensemble_prediction': ensemble_prediction,
                'individual_predictions': predictions,
                'confidences': confidences,
                'clinical_prediction': clinical_prediction,
                'clinical_validation': clinical_validation,
                'feature_importance': self._get_feature_importance(),
                'classification_confidence': ensemble_prediction.get('confidence', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            # Fallback to clinical classification
            try:
                clinical = self._clinical_classification(features)
                return {
                    'ensemble_prediction': clinical,
                    'error': str(e),
                    'method_used': 'clinical_fallback'
                }
            except:
                return {
                    'ensemble_prediction': {'class': 'neutral', 'confidence': 0.0},
                    'error': str(e)
                }
    
    def _ensemble_predict(self, predictions: Dict[str, str], confidences: Dict[str, float]) -> Dict[str, Any]:
        """Combine predictions from multiple models"""
        if not predictions:
            return {'class': 'neutral', 'confidence': 0.0}
        
        # Weight votes by confidence
        class_votes = {}
        total_weight = 0
        
        for model, pred_class in predictions.items():
            confidence = confidences.get(model, 0.5)
            if pred_class not in class_votes:
                class_votes[pred_class] = 0
            class_votes[pred_class] += confidence
            total_weight += confidence
        
        # Normalize votes
        if total_weight > 0:
            for class_name in class_votes:
                class_votes[class_name] /= total_weight
        
        # Select class with highest weighted vote
        ensemble_class = max(class_votes, key=class_votes.get)
        ensemble_confidence = class_votes[ensemble_class]
        
        return {
            'class': ensemble_class,
            'confidence': float(ensemble_confidence),
            'class_probabilities': class_votes
        }
    
    def _clinical_classification(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clinical classification using validated thresholds from research
        More reliable than ML with dummy training data
        """
        try:
            # Use clinical classifier if available
            if self.clinical_classifier:
                raw_features = features.get('raw_features', {})
                biomech = raw_features.get('biomechanical', [])
                
                if len(biomech) >= 4:
                    left_eversion_mean = biomech[0]
                    right_eversion_mean = biomech[2]
                    
                    # Classify each foot individually
                    left_class, left_conf = self.clinical_classifier.classify_pronation(left_eversion_mean)
                    right_class, right_conf = self.clinical_classifier.classify_pronation(right_eversion_mean)
                    
                    # Debug logging
                    logger.info(f"Clinical classification debug:")
                    logger.info(f"  Left foot: {left_eversion_mean}° → {left_class} (confidence: {left_conf:.3f})")
                    logger.info(f"  Right foot: {right_eversion_mean}° → {right_class} (confidence: {right_conf:.3f})")
                    
                    left_severity = self._get_severity_score(left_class)
                    right_severity = self._get_severity_score(right_class)
                    logger.info(f"  Severity scores: Left={left_severity}, Right={right_severity}")
                    
                    # Determine overall classification based on more severe case
                    if left_severity > right_severity:
                        primary_class = left_class
                        primary_confidence = left_conf
                        primary_angle = left_eversion_mean
                        logger.info(f"  Selected: Left foot ({left_class})")
                    else:
                        primary_class = right_class
                        primary_confidence = right_conf
                        primary_angle = right_eversion_mean
                        logger.info(f"  Selected: Right foot ({right_class})")
                    
                    # Reduce confidence if there's significant asymmetry
                    asymmetry = abs(left_eversion_mean - right_eversion_mean)
                    if asymmetry > 10:  # High asymmetry
                        primary_confidence *= 0.8
                    
                    return {
                        'class': primary_class.replace('_', ' ').title() if '_' in primary_class else primary_class,
                        'confidence': float(primary_confidence),
                        'primary_angle': float(primary_angle),
                        'left_angle': float(left_eversion_mean),
                        'right_angle': float(right_eversion_mean),
                        'asymmetry': float(asymmetry),
                        'method': 'clinical_validated_individual'
                    }
            
            # Fallback to built-in clinical thresholds
            return self._rule_based_classification(features)
            
        except Exception as e:
            logger.error(f"Clinical classification error: {str(e)}")
            return {'class': 'neutral', 'confidence': 0.5, 'method': 'default'}
    
    def _get_severity_score(self, classification: str) -> int:
        """Get severity score for classification to determine which foot is worse"""
        severity_scores = {
            'neutral': 0,
            'mild_overpronation': 2,      # Updated to match new classification names
            'moderate_overpronation': 4,  # Higher priority than supination
            'severe_overpronation': 6,    # Highest priority
            'mild_pronation': 2,          # Support both naming conventions
            'moderate_pronation': 4,
            'severe_pronation': 6,
            'supination': 1               # Lower priority than pronation issues
        }
        return severity_scores.get(classification.replace(' ', '_').lower(), 0)
    
    def _validate_with_clinical_thresholds(self, classification: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate ML classification against clinical thresholds
        """
        try:
            raw_features = features.get('raw_features', {})
            biomech = raw_features.get('biomechanical', [])
            
            if len(biomech) >= 4:
                left_eversion = biomech[0]
                right_eversion = biomech[2]
                avg_eversion = (left_eversion + right_eversion) / 2
                
                # Check if classification matches clinical thresholds
                classification_key = classification.lower().replace(' ', '_')
                if classification_key in self.clinical_thresholds['pronation']:
                    min_val, max_val = self.clinical_thresholds['pronation'][classification_key]
                    
                    if min_val == float('-inf'):
                        is_valid = avg_eversion < max_val
                    elif max_val == float('inf'):
                        is_valid = avg_eversion >= min_val
                    else:
                        is_valid = min_val <= avg_eversion <= max_val
                    
                    return {
                        'is_valid': is_valid,
                        'confidence': 1.0 if is_valid else 0.3,
                        'expected_range': (min_val, max_val),
                        'actual_value': avg_eversion
                    }
            
            return {'is_valid': True, 'confidence': 0.5, 'note': 'Insufficient data for validation'}
            
        except Exception as e:
            logger.error(f"Clinical validation error: {str(e)}")
            return {'is_valid': True, 'confidence': 0.5, 'error': str(e)}
    
    def _rule_based_classification(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based classification as backup/validation
        Based on biomechanical thresholds
        """
        try:
            raw_features = features.get('raw_features', {})
            biomech = raw_features.get('biomechanical', [])
            
            if len(biomech) >= 4:
                left_eversion_mean = biomech[0]
                right_eversion_mean = biomech[2]
                avg_eversion = (left_eversion_mean + right_eversion_mean) / 2
                
                # Biomechanical thresholds (degrees) 
                # Note: Negative angles = pronation (inward roll), Positive angles = supination (outward roll)
                if -5 <= avg_eversion <= 5:
                    rule_class = 'neutral'
                elif -10 <= avg_eversion < -5:
                    rule_class = 'mild_pronation'
                elif -15 <= avg_eversion < -10:
                    rule_class = 'moderate_pronation'
                elif avg_eversion < -15:
                    rule_class = 'severe_pronation'
                else:  # avg_eversion > 5
                    rule_class = 'supination'
                
                # Calculate confidence based on how far from threshold boundaries
                confidence = self._calculate_rule_confidence(avg_eversion, rule_class)
                
                return {
                    'class': rule_class,
                    'confidence': confidence,
                    'avg_eversion_angle': float(avg_eversion)
                }
            
            return {'class': 'neutral', 'confidence': 0.0}
            
        except Exception as e:
            logger.error(f"Rule-based classification error: {str(e)}")
            return {'class': 'neutral', 'confidence': 0.0}
    
    def _calculate_rule_confidence(self, angle: float, predicted_class: str) -> float:
        """Calculate confidence for rule-based classification"""
        thresholds = {
            'neutral': (-5, 5),
            'mild_pronation': (-10, -5),
            'moderate_pronation': (-15, -10),
            'severe_pronation': (float('-inf'), -15),
            'supination': (5, float('inf'))
        }
        
        if predicted_class in thresholds:
            min_thresh, max_thresh = thresholds[predicted_class]
            
            # Calculate distance from boundaries
            if min_thresh == float('-inf'):
                distance = max_thresh - angle
            elif max_thresh == float('inf'):
                distance = angle - min_thresh
            else:
                center = (min_thresh + max_thresh) / 2
                distance = (max_thresh - min_thresh) / 2 - abs(angle - center)
            
            # Convert to confidence (0.5-1.0 range)
            confidence = 0.5 + min(distance / 10, 0.5)
            return max(0.5, min(1.0, confidence))
        
        return 0.5
    
    async def retrain(self) -> Dict[str, Any]:
        """
        Retrain all models with available training data
        """
        try:
            # Load training data
            X, y = self._load_training_data()
            
            if len(X) < 10:  # Minimum samples required
                return {
                    'error': 'Insufficient training data',
                    'samples_required': 10,
                    'samples_available': len(X)
                }
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            metrics = {}
            
            # Train Random Forest
            self.rf_model.fit(X_train_scaled, y_train)
            rf_pred = self.rf_model.predict(X_test_scaled)
            metrics['random_forest'] = {
                'accuracy': float(accuracy_score(y_test, rf_pred)),
                'classification_report': classification_report(y_test, rf_pred, output_dict=True)
            }
            
            # Train Gradient Boosting
            self.gb_model.fit(X_train_scaled, y_train)
            gb_pred = self.gb_model.predict(X_test_scaled)
            metrics['gradient_boosting'] = {
                'accuracy': float(accuracy_score(y_test, gb_pred)),
                'classification_report': classification_report(y_test, gb_pred, output_dict=True)
            }
            
            # Train Neural Network
            nn_metrics = self._train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test)
            metrics['neural_network'] = nn_metrics
            
            # Save models
            self._save_models()
            
            # Update performance metrics
            self.performance_metrics = metrics
            
            logger.info("Model retraining completed successfully")
            return {
                'status': 'success',
                'metrics': metrics,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            logger.error(f"Retraining error: {str(e)}")
            return {'error': str(e)}
    
    def _train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray, 
                            X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train the neural network model"""
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        
        # Initialize model
        input_size = X_train.shape[1]
        self.neural_model = NeuralGaitClassifier(input_size=input_size)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.neural_model.parameters(), lr=0.001)
        
        # Training loop
        epochs = 100
        best_loss = float('inf')
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.neural_model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
        
        # Evaluate
        with torch.no_grad():
            test_outputs = self.neural_model(X_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == torch.LongTensor(y_test)).float().mean()
        
        return {
            'accuracy': float(accuracy),
            'final_loss': float(best_loss),
            'epochs': epochs
        }
    
    def _load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data from datasets"""
        # This would load from your dataset files
        # For now, return dummy data structure
        
        data_file = self.model_dir.parent / "datasets" / "training_data.json"
        
        if data_file.exists():
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            X = np.array([sample['features'] for sample in data])
            y = np.array([self.class_labels.index(sample['label']) for sample in data])
            
            return X, y
        else:
            # Return dummy data for initial setup
            np.random.seed(42)
            X = np.random.rand(100, 20)  # 100 samples, 20 features
            y = np.random.randint(0, len(self.class_labels), 100)
            
            return X, y
    
    def _save_models(self):
        """Save trained models"""
        try:
            # Save scikit-learn models
            joblib.dump(self.rf_model, self.model_dir / "random_forest.pkl")
            joblib.dump(self.gb_model, self.model_dir / "gradient_boosting.pkl")
            joblib.dump(self.scaler, self.model_dir / "scaler.pkl")
            
            # Save PyTorch model
            torch.save(self.neural_model.state_dict(), self.model_dir / "neural_network.pth")
            
            # Save metadata
            metadata = {
                'class_labels': self.class_labels,
                'feature_size': 20,
                'model_version': '2.0.0'
            }
            
            with open(self.model_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            rf_path = self.model_dir / "random_forest.pkl"
            gb_path = self.model_dir / "gradient_boosting.pkl"
            scaler_path = self.model_dir / "scaler.pkl"
            nn_path = self.model_dir / "neural_network.pth"
            
            models_loaded = False
            
            if rf_path.exists():
                self.rf_model = joblib.load(rf_path)
                logger.info("Random Forest model loaded")
                models_loaded = True
            
            if gb_path.exists():
                self.gb_model = joblib.load(gb_path)
                logger.info("Gradient Boosting model loaded")
                models_loaded = True
            
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded")
            else:
                # Initialize scaler with dummy data if not available
                logger.info("Initializing scaler with default data")
                dummy_data = np.random.rand(100, 20)
                self.scaler.fit(dummy_data)
            
            if nn_path.exists():
                self.neural_model.load_state_dict(torch.load(nn_path))
                logger.info("Neural Network model loaded")
                models_loaded = True
            
            # If no models are loaded, train with dummy data to initialize
            if not models_loaded:
                logger.info("No pre-trained models found, initializing with dummy data")
                self._initialize_with_dummy_data()
            
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {str(e)}")
            self._initialize_with_dummy_data()
    
    def _initialize_with_dummy_data(self):
        """Initialize models with dummy data for first-time setup"""
        try:
            # Generate dummy training data
            np.random.seed(42)
            X_dummy = np.random.rand(100, 20)
            y_dummy = np.random.randint(0, len(self.class_labels), 100)
            
            # Fit scaler
            self.scaler.fit(X_dummy)
            X_scaled = self.scaler.transform(X_dummy)
            
            # Train models with dummy data
            self.rf_model.fit(X_scaled, y_dummy)
            self.gb_model.fit(X_scaled, y_dummy)
            
            logger.info("Models initialized with dummy data")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        return {
            'metrics': self.performance_metrics,
            'model_info': {
                'random_forest_trees': getattr(self.rf_model, 'n_estimators', 0),
                'gradient_boosting_estimators': getattr(self.gb_model, 'n_estimators', 0),
                'neural_network_params': sum(p.numel() for p in self.neural_model.parameters()),
                'class_labels': self.class_labels
            },
            'training_status': 'trained' if self.performance_metrics else 'not_trained'
        }
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained models"""
        importance = {}
        
        feature_names = [
            'left_eversion_mean', 'left_eversion_std', 'right_eversion_mean', 'right_eversion_std',
            'eversion_asymmetry', 'cadence', 'step_duration', 'avg_step_length', 'step_variability'
        ]
        
        if hasattr(self.rf_model, 'feature_importances_'):
            rf_importance = self.rf_model.feature_importances_
            importance['random_forest'] = dict(zip(feature_names[:len(rf_importance)], 
                                                 rf_importance.tolist()))
        
        return importance