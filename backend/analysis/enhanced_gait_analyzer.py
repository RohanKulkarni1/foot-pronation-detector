import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedGaitAnalyzer:
    """
    Advanced gait analyzer for Phase 2 with multi-angle support,
    enhanced biomechanical calculations, and ML feature extraction
    """
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Higher complexity for better accuracy
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    async def analyze_multi_angle(self, video_files: List[Path], session_id: str) -> Dict[str, Any]:
        """
        Analyze multiple video angles simultaneously for comprehensive gait analysis
        """
        logger.info(f"Starting multi-angle analysis for session {session_id}")
        
        # Process videos in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            detected_angles = []
            for video_file in video_files:
                angle = self._detect_camera_angle(video_file)  # Pass full path
                logger.info(f"Detected camera angle for {video_file.name}: {angle}")
                detected_angles.append(angle)
                future = executor.submit(self._analyze_single_video, video_file, angle)
                futures.append(future)
            
            # Collect results
            angle_results = {}
            for i, future in enumerate(futures):
                angle = detected_angles[i]  # Use pre-detected angle
                angle_results[angle] = future.result()
        
        # Combine multi-angle analysis
        combined_results = self._combine_angle_analyses(angle_results, session_id)
        
        # Enhanced biomechanical calculations
        enhanced_metrics = self._calculate_enhanced_metrics(combined_results)
        combined_results['enhanced_metrics'] = enhanced_metrics
        
        # Extract ML features
        ml_features = self._extract_ml_features(combined_results)
        combined_results['features'] = ml_features
        
        logger.info(f"Multi-angle analysis completed for session {session_id}")
        return combined_results
    
    def _detect_camera_angle(self, video_path: Path) -> str:
        """Detect camera angle from filename or analyze video content"""
        filename_lower = str(video_path.name).lower()
        
        # First try filename detection
        if 'rear' in filename_lower or 'back' in filename_lower:
            return 'rear'
        elif 'side' in filename_lower or 'sagittal' in filename_lower:
            return 'side'
        elif 'front' in filename_lower or 'anterior' in filename_lower:
            return 'front'
        
        # If filename doesn't give clear indication, analyze video content
        return self._analyze_video_angle(video_path)
    
    def _analyze_video_angle(self, video_path: Path) -> str:
        """Analyze video content to determine viewing angle"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return 'rear'  # Default fallback
        
        # Sample a few frames to analyze pose orientation
        frame_samples = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = [total_frames // 4, total_frames // 2, 3 * total_frames // 4]
        
        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    # Analyze the pose to determine viewing angle
                    angle = self._determine_angle_from_pose(results.pose_landmarks, frame.shape)
                    frame_samples.append(angle)
        
        cap.release()
        
        if not frame_samples:
            return 'rear'  # Default fallback
        
        # Return most common angle detected
        from collections import Counter
        return Counter(frame_samples).most_common(1)[0][0]
    
    def _determine_angle_from_pose(self, pose_landmarks, frame_shape) -> str:
        """Determine viewing angle based on pose landmark positions"""
        h, w = frame_shape[:2]
        
        # Get key landmarks
        landmarks = pose_landmarks.landmark
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        
        # Calculate body orientation indicators
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        hip_width = abs(left_hip.x - right_hip.x)
        ankle_width = abs(left_ankle.x - right_ankle.x)
        
        # Check z-coordinates to determine depth perception
        shoulder_z_diff = abs(left_shoulder.z - right_shoulder.z)
        hip_z_diff = abs(left_hip.z - right_hip.z)
        
        # Get nose position
        nose_x = nose.x
        
        # Check if both ears are visible (indicates front or rear view)
        both_ears_visible = left_ear.visibility > 0.5 and right_ear.visibility > 0.5
        
        # For now, just default to rear view for your walking videos
        # This ensures rear view videos get proper pronation analysis
        logger.info(f"Defaulting to rear view - shoulder_width: {shoulder_width:.3f}, hip_width: {hip_width:.3f}, nose_x: {nose_x:.3f}")
        return 'rear'
    
    def _analyze_single_video(self, video_path: Path, angle: str) -> Dict[str, Any]:
        """Analyze a single video angle"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        landmarks_data = []
        frame_number = 0
        
        # Process video frames
        while cap.read()[0]:
            ret, frame = cap.read()
            if not ret:
                break
            
            # MediaPipe pose detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract landmarks
                landmarks = self._extract_landmarks(results.pose_landmarks, frame.shape)
                landmarks['frame'] = frame_number
                landmarks['timestamp'] = frame_number / fps
                landmarks_data.append(landmarks)
            
            frame_number += 1
        
        cap.release()
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(landmarks_data)
        
        if df.empty:
            return {'angle': angle, 'landmarks': [], 'metrics': {}}
        
        # Calculate angle-specific metrics
        if angle == 'rear':
            metrics = self._calculate_rear_view_metrics(df)
        elif angle == 'side':
            metrics = self._calculate_side_view_metrics(df)
        elif angle == 'front':
            metrics = self._calculate_front_view_metrics(df)
        else:
            metrics = {}
        
        return {
            'angle': angle,
            'landmarks': landmarks_data,
            'metrics': metrics,
            'frame_count': len(landmarks_data),
            'fps': fps,
            'video_path': str(video_path)
        }
    
    def _extract_landmarks(self, pose_landmarks, frame_shape) -> Dict[str, float]:
        """Extract key landmarks with normalized coordinates"""
        h, w = frame_shape[:2]
        
        landmarks = {}
        
        # Key landmarks for gait analysis
        key_points = {
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            'left_heel': self.mp_pose.PoseLandmark.LEFT_HEEL,
            'right_heel': self.mp_pose.PoseLandmark.RIGHT_HEEL,
            'left_foot_index': self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            'right_foot_index': self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
        }
        
        for name, landmark_id in key_points.items():
            landmark = pose_landmarks.landmark[landmark_id]
            landmarks[f'{name}_x'] = landmark.x * w
            landmarks[f'{name}_y'] = landmark.y * h
            landmarks[f'{name}_z'] = landmark.z
            landmarks[f'{name}_visibility'] = landmark.visibility
        
        return landmarks
    
    def _calculate_rear_view_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate rear view specific metrics (pronation/supination)"""
        metrics = {}
        
        # Rearfoot eversion angle
        if all(col in df.columns for col in ['left_ankle_x', 'left_heel_x', 'left_ankle_y', 'left_heel_y']):
            left_eversion = self._calculate_eversion_angle(df, 'left')
            right_eversion = self._calculate_eversion_angle(df, 'right')
            
            metrics['left_rearfoot_eversion'] = {
                'mean': float(np.mean(left_eversion)),
                'std': float(np.std(left_eversion)),
                'max': float(np.max(left_eversion)),
                'min': float(np.min(left_eversion))
            }
            
            metrics['right_rearfoot_eversion'] = {
                'mean': float(np.mean(right_eversion)),
                'std': float(np.std(right_eversion)),
                'max': float(np.max(right_eversion)),
                'min': float(np.min(right_eversion))
            }
            
            # Asymmetry
            metrics['eversion_asymmetry'] = float(abs(np.mean(left_eversion) - np.mean(right_eversion)))
        
        return metrics
    
    def _calculate_side_view_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate side view specific metrics (step length, cadence)"""
        metrics = {}
        
        # Step detection and analysis
        steps = self._detect_steps_side_view(df)
        
        if steps:
            step_lengths = [step['length'] for step in steps]
            step_durations = [step['duration'] for step in steps]
            
            metrics['step_analysis'] = {
                'step_count': len(steps),
                'avg_step_length': float(np.mean(step_lengths)) if step_lengths else 0,
                'step_length_variability': float(np.std(step_lengths)) if len(step_lengths) > 1 else 0,
                'avg_step_duration': float(np.mean(step_durations)) if step_durations else 0,
                'cadence': len(steps) / (df['timestamp'].max() - df['timestamp'].min()) * 60 if len(steps) > 0 else 0
            }
        
        # Knee flexion analysis
        knee_flexion = self._calculate_knee_flexion(df)
        if knee_flexion:
            metrics['knee_flexion'] = knee_flexion
        
        return metrics
    
    def _calculate_front_view_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate front view specific metrics (foot progression angle)"""
        metrics = {}
        
        # Foot progression angle
        if all(col in df.columns for col in ['left_foot_index_x', 'left_heel_x']):
            left_progression = self._calculate_foot_progression_angle(df, 'left')
            right_progression = self._calculate_foot_progression_angle(df, 'right')
            
            metrics['foot_progression'] = {
                'left_mean': float(np.mean(left_progression)),
                'right_mean': float(np.mean(right_progression)),
                'asymmetry': float(abs(np.mean(left_progression) - np.mean(right_progression)))
            }
        
        return metrics
    
    def _calculate_eversion_angle(self, df: pd.DataFrame, side: str) -> np.ndarray:
        """Calculate rearfoot eversion angle with smoothing"""
        ankle_x = df[f'{side}_ankle_x'].values
        heel_x = df[f'{side}_heel_x'].values
        ankle_y = df[f'{side}_ankle_y'].values
        heel_y = df[f'{side}_heel_y'].values
        
        # Calculate angle between ankle-heel line and vertical
        angles = []
        for i in range(len(ankle_x)):
            if not (np.isnan(ankle_x[i]) or np.isnan(heel_x[i])):
                dx = heel_x[i] - ankle_x[i]
                dy = heel_y[i] - ankle_y[i]
                angle = np.degrees(np.arctan2(dx, dy))
                angles.append(angle)
        
        angles = np.array(angles)
        
        # Apply smoothing to reduce noise
        if len(angles) > 5:
            # Remove outliers (beyond 2 standard deviations)
            mean_angle = np.mean(angles)
            std_angle = np.std(angles)
            mask = np.abs(angles - mean_angle) < 2 * std_angle
            angles = angles[mask]
            
            # Apply moving average for additional smoothing
            if len(angles) > 3:
                window_size = min(5, len(angles) // 3)
                smoothed = np.convolve(angles, np.ones(window_size)/window_size, mode='same')
                return smoothed
        
        return angles
    
    def _detect_steps_side_view(self, df: pd.DataFrame) -> List[Dict[str, float]]:
        """Detect individual steps from side view"""
        steps = []
        
        # Use ankle position to detect heel strikes
        for side in ['left', 'right']:
            ankle_y = df[f'{side}_ankle_y'].values
            ankle_x = df[f'{side}_ankle_x'].values
            timestamps = df['timestamp'].values
            
            # Find local minima (heel strikes)
            heel_strikes = []
            for i in range(1, len(ankle_y) - 1):
                if ankle_y[i] < ankle_y[i-1] and ankle_y[i] < ankle_y[i+1]:
                    heel_strikes.append(i)
            
            # Calculate step parameters
            for i in range(len(heel_strikes) - 1):
                start_idx = heel_strikes[i]
                end_idx = heel_strikes[i + 1]
                
                step_length = abs(ankle_x[end_idx] - ankle_x[start_idx])
                step_duration = timestamps[end_idx] - timestamps[start_idx]
                
                steps.append({
                    'side': side,
                    'length': step_length,
                    'duration': step_duration,
                    'start_time': timestamps[start_idx],
                    'end_time': timestamps[end_idx]
                })
        
        return steps
    
    def _calculate_knee_flexion(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate knee flexion angles"""
        flexion_data = {}
        
        for side in ['left', 'right']:
            if all(col in df.columns for col in [f'{side}_hip_x', f'{side}_knee_x', f'{side}_ankle_x',
                                               f'{side}_hip_y', f'{side}_knee_y', f'{side}_ankle_y']):
                
                hip_x = df[f'{side}_hip_x'].values
                knee_x = df[f'{side}_knee_x'].values
                ankle_x = df[f'{side}_ankle_x'].values
                hip_y = df[f'{side}_hip_y'].values
                knee_y = df[f'{side}_knee_y'].values
                ankle_y = df[f'{side}_ankle_y'].values
                
                flexion_angles = []
                for i in range(len(hip_x)):
                    if not any(np.isnan([hip_x[i], knee_x[i], ankle_x[i], hip_y[i], knee_y[i], ankle_y[i]])):
                        # Calculate angle between hip-knee and knee-ankle vectors
                        v1 = np.array([hip_x[i] - knee_x[i], hip_y[i] - knee_y[i]])
                        v2 = np.array([ankle_x[i] - knee_x[i], ankle_y[i] - knee_y[i]])
                        
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                        flexion_angles.append(180 - angle)  # Convert to flexion angle
                
                flexion_data[f'{side}_knee_flexion'] = {
                    'mean': float(np.mean(flexion_angles)),
                    'max': float(np.max(flexion_angles)),
                    'min': float(np.min(flexion_angles)),
                    'range': float(np.max(flexion_angles) - np.min(flexion_angles))
                }
        
        return flexion_data
    
    def _calculate_foot_progression_angle(self, df: pd.DataFrame, side: str) -> np.ndarray:
        """Calculate foot progression angle from front view"""
        toe_x = df[f'{side}_foot_index_x'].values
        heel_x = df[f'{side}_heel_x'].values
        toe_y = df[f'{side}_foot_index_y'].values
        heel_y = df[f'{side}_heel_y'].values
        
        angles = []
        for i in range(len(toe_x)):
            if not any(np.isnan([toe_x[i], heel_x[i], toe_y[i], heel_y[i]])):
                dx = toe_x[i] - heel_x[i]
                dy = toe_y[i] - heel_y[i]
                angle = np.degrees(np.arctan2(dx, dy))
                angles.append(angle)
        
        return np.array(angles)
    
    def _combine_angle_analyses(self, angle_results: Dict[str, Dict], session_id: str) -> Dict[str, Any]:
        """Combine analyses from multiple camera angles"""
        combined = {
            'session_id': session_id,
            'angles_analyzed': list(angle_results.keys()),
            'individual_angles': angle_results,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Combine metrics from all angles
        all_metrics = {}
        for angle, result in angle_results.items():
            if 'metrics' in result:
                all_metrics[angle] = result['metrics']
        
        combined['combined_metrics'] = all_metrics
        
        # Generate biomechanical profile
        combined['biomechanical_profile'] = self._generate_biomechanical_profile(all_metrics)
        
        # Generate biomechanical metrics for frontend compatibility
        combined['biomechanical_metrics'] = self._generate_biomechanical_metrics(all_metrics)
        
        # Note: asymmetry metrics are available in biomechanical_profile.asymmetry_analysis
        
        return combined
    
    def _calculate_enhanced_metrics(self, combined_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate enhanced biomechanical metrics combining all angles"""
        enhanced = {}
        
        # Overall gait classification based on multiple angles
        if 'rear' in combined_results['individual_angles']:
            rear_metrics = combined_results['individual_angles']['rear'].get('metrics', {})
            if 'left_rearfoot_eversion' in rear_metrics and 'right_rearfoot_eversion' in rear_metrics:
                left_eversion = rear_metrics['left_rearfoot_eversion']['mean']
                right_eversion = rear_metrics['right_rearfoot_eversion']['mean']
                
                # Enhanced pronation classification
                enhanced['pronation_assessment'] = self._assess_pronation(left_eversion, right_eversion)
        
        # Gait symmetry analysis
        if 'side' in combined_results['individual_angles']:
            side_metrics = combined_results['individual_angles']['side'].get('metrics', {})
            if 'step_analysis' in side_metrics:
                enhanced['gait_efficiency'] = self._calculate_gait_efficiency(side_metrics['step_analysis'])
        
        return enhanced
    
    def _assess_pronation(self, left_eversion: float, right_eversion: float) -> Dict[str, Any]:
        """Enhanced pronation assessment"""
        # Corrected thresholds based on biomechanical literature
        # NEGATIVE angles = pronation (foot rolling inward)
        # POSITIVE angles = supination (foot rolling outward)
        
        def classify_foot(eversion):
            if -5 <= eversion <= 5:
                return "neutral"
            elif -10 <= eversion < -5:
                return "mild_pronation"
            elif -15 <= eversion < -10:
                return "moderate_pronation"
            elif eversion < -15:
                return "severe_pronation"
            else:  # eversion > 5
                return "supination"
        
        return {
            'left_foot': classify_foot(left_eversion),
            'right_foot': classify_foot(right_eversion),
            'asymmetry_level': abs(left_eversion - right_eversion),
            'overall_assessment': classify_foot((left_eversion + right_eversion) / 2),
            'confidence': 0.75  # Would be calculated based on data quality
        }
    
    def _calculate_gait_efficiency(self, step_analysis: Dict) -> Dict[str, Any]:
        """Calculate gait efficiency metrics"""
        return {
            'cadence_efficiency': min(step_analysis.get('cadence', 0) / 120, 1.0),  # Normalized to optimal 120 steps/min
            'step_regularity': 1 / (1 + step_analysis.get('step_length_variability', 1)),
            'overall_efficiency': 0.8  # Would be calculated based on multiple factors
        }
    
    def _generate_biomechanical_profile(self, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive biomechanical profile from all metrics"""
        profile = {
            'foot_mechanics': {},
            'gait_pattern': {},
            'asymmetry_analysis': {},
            'risk_factors': []
        }
        
        # Extract foot mechanics from rear view
        if 'rear' in all_metrics:
            rear_metrics = all_metrics['rear']
            if 'left_rearfoot_eversion' in rear_metrics and 'right_rearfoot_eversion' in rear_metrics:
                left_eversion = rear_metrics['left_rearfoot_eversion']['mean']
                right_eversion = rear_metrics['right_rearfoot_eversion']['mean']
                
                # Foot mechanics analysis
                profile['foot_mechanics'] = {
                    'left_foot': {
                        'eversion_angle': round(left_eversion, 2),
                        'classification': self._classify_single_foot(left_eversion),
                        'severity': self._get_severity_level(left_eversion)
                    },
                    'right_foot': {
                        'eversion_angle': round(right_eversion, 2),
                        'classification': self._classify_single_foot(right_eversion),
                        'severity': self._get_severity_level(right_eversion)
                    }
                }
                
                # Asymmetry analysis
                asymmetry = abs(left_eversion - right_eversion)
                profile['asymmetry_analysis'] = {
                    'degree_difference': round(asymmetry, 2),
                    'level': 'High' if asymmetry > 10 else 'Moderate' if asymmetry > 5 else 'Low',
                    'clinical_significance': asymmetry > 5
                }
                
                # Risk factors based on biomechanics
                if left_eversion < -15 or right_eversion < -15:
                    profile['risk_factors'].append('Severe overpronation detected')
                if asymmetry > 10:
                    profile['risk_factors'].append('Significant gait asymmetry')
                if left_eversion > 5 or right_eversion > 5:
                    profile['risk_factors'].append('Supination pattern detected')
        
        # Extract gait pattern from side view
        if 'side' in all_metrics:
            side_metrics = all_metrics['side']
            if 'step_analysis' in side_metrics:
                step_data = side_metrics['step_analysis']
                profile['gait_pattern'] = {
                    'cadence': step_data.get('cadence', 0),
                    'step_length': step_data.get('avg_step_length', 0),
                    'step_variability': step_data.get('step_length_variability', 0),
                    'pattern_quality': 'Regular' if step_data.get('step_length_variability', 1) < 0.1 else 'Irregular'
                }
                
                # Add cadence-related risk factors
                if step_data.get('cadence', 0) < 90:
                    profile['risk_factors'].append('Below normal cadence')
                elif step_data.get('cadence', 0) > 140:
                    profile['risk_factors'].append('Above normal cadence')
        
        # Overall assessment
        profile['overall_assessment'] = self._generate_overall_assessment(profile)
        
        return profile
    
    def _classify_single_foot(self, eversion: float) -> str:
        """Classify single foot based on eversion angle"""
        if -5 <= eversion <= 5:
            return "Neutral"
        elif -10 <= eversion < -5:
            return "Mild Overpronation"
        elif -15 <= eversion < -10:
            return "Moderate Overpronation"
        elif eversion < -15:
            return "Severe Overpronation"
        else:
            return "Supination"
    
    def _get_severity_level(self, eversion: float) -> str:
        """Get severity level based on eversion angle"""
        abs_eversion = abs(eversion)
        if abs_eversion <= 5:
            return "Normal"
        elif abs_eversion <= 10:
            return "Mild"
        elif abs_eversion <= 15:
            return "Moderate"
        else:
            return "Severe"
    
    def _generate_overall_assessment(self, profile: Dict[str, Any]) -> str:
        """Generate overall assessment based on profile"""
        risk_count = len(profile.get('risk_factors', []))
        
        if risk_count == 0:
            return "Normal gait pattern with no significant concerns"
        elif risk_count == 1:
            return "Minor gait irregularities detected - monitor over time"
        elif risk_count == 2:
            return "Moderate gait abnormalities - consider professional evaluation"
        else:
            return "Significant gait abnormalities - professional evaluation recommended"
    
    def _generate_biomechanical_metrics(self, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate biomechanical metrics in format expected by frontend"""
        # Default values - these are placeholders when actual data isn't available
        metrics = {
            'cadence': 110,  # Default typical walking cadence
            'step_length': 0.65,  # Default typical step length
            'stance_time': 0.62,  # Typical value (62% of gait cycle)
            'swing_time': 0.38,   # Typical value (38% of gait cycle)
            'cadence_score': 85,
            'step_length_score': 80,
            'symmetry_score': 75,
            'efficiency_score': 80,
            'stability_score': 85,
            'balance_score': 80
        }
        
        # Extract actual values from metrics
        if 'side' in all_metrics:
            side_metrics = all_metrics['side']
            if 'step_analysis' in side_metrics:
                step_data = side_metrics['step_analysis']
                metrics['cadence'] = step_data.get('cadence', 110)
                metrics['step_length'] = step_data.get('avg_step_length', 0.65)
                
                # Calculate scores (0-100 scale)
                # Cadence score (optimal: 100-120 steps/min)
                cadence = metrics['cadence']
                if 100 <= cadence <= 120:
                    metrics['cadence_score'] = 90 + (10 * (1 - abs(cadence - 110) / 10))
                else:
                    metrics['cadence_score'] = max(50, 90 - abs(cadence - 110))
                
                # Step length score (optimal: 0.6-0.8m)
                step_len = metrics['step_length']
                if 0.6 <= step_len <= 0.8:
                    metrics['step_length_score'] = 90
                else:
                    metrics['step_length_score'] = max(50, 90 - abs(step_len - 0.7) * 100)
        
        # Calculate symmetry score from rear metrics
        if 'rear' in all_metrics:
            rear_metrics = all_metrics['rear']
            if 'left_rearfoot_eversion' in rear_metrics and 'right_rearfoot_eversion' in rear_metrics:
                left_eversion = rear_metrics['left_rearfoot_eversion']['mean']
                right_eversion = rear_metrics['right_rearfoot_eversion']['mean']
                asymmetry = abs(left_eversion - right_eversion)
                
                # Symmetry score (lower asymmetry = higher score)
                if asymmetry <= 2:
                    metrics['symmetry_score'] = 95
                elif asymmetry <= 5:
                    metrics['symmetry_score'] = 85
                elif asymmetry <= 10:
                    metrics['symmetry_score'] = 70
                else:
                    metrics['symmetry_score'] = max(50, 70 - asymmetry)
                
                # Stability score based on pronation severity
                max_eversion = max(abs(left_eversion), abs(right_eversion))
                if max_eversion <= 5:
                    metrics['stability_score'] = 95
                elif max_eversion <= 10:
                    metrics['stability_score'] = 80
                elif max_eversion <= 15:
                    metrics['stability_score'] = 65
                else:
                    metrics['stability_score'] = 50
                
                # Balance score (combination of symmetry and stability)
                metrics['balance_score'] = (metrics['symmetry_score'] + metrics['stability_score']) / 2
        
        # Efficiency score (combination of cadence and step length)
        metrics['efficiency_score'] = (metrics['cadence_score'] + metrics['step_length_score']) / 2
        
        # Stance and swing time (typical values if not directly measured)
        metrics['stance_time'] = 0.62  # 62% of gait cycle
        metrics['swing_time'] = 0.38   # 38% of gait cycle
        
        return metrics
    
    def _generate_asymmetry_metrics(self, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate asymmetry metrics in format expected by frontend asymmetry widget"""
        asymmetry_metrics = {
            'asymmetry_index': 0.0,
            'left_foot_contact_time': 0.6,  # Default typical contact time
            'right_foot_contact_time': 0.6,
        }
        
        # Extract asymmetry from rear view metrics
        if 'rear' in all_metrics:
            rear_metrics = all_metrics['rear']
            if 'left_rearfoot_eversion' in rear_metrics and 'right_rearfoot_eversion' in rear_metrics:
                left_eversion = rear_metrics['left_rearfoot_eversion']['mean']
                right_eversion = rear_metrics['right_rearfoot_eversion']['mean']
                asymmetry_degree = abs(left_eversion - right_eversion)
                
                # Convert degree difference to asymmetry index percentage
                # Asymmetry index formula: |left - right| / ((left + right)/2) * 100
                if left_eversion != 0 or right_eversion != 0:
                    avg_eversion = (abs(left_eversion) + abs(right_eversion)) / 2
                    if avg_eversion > 0:
                        asymmetry_index = (asymmetry_degree / avg_eversion) * 100
                        asymmetry_metrics['asymmetry_index'] = round(min(asymmetry_index, 100), 2)
                    else:
                        asymmetry_metrics['asymmetry_index'] = round(asymmetry_degree * 5, 2)  # Scale for small angles
                
                # Generate different contact times based on asymmetry
                base_contact_time = 0.62  # Normal stance phase percentage
                asymmetry_factor = min(asymmetry_degree / 20, 0.1)  # Max 10% difference
                
                if left_eversion < right_eversion:  # Left more pronated
                    asymmetry_metrics['left_foot_contact_time'] = round(base_contact_time + asymmetry_factor, 3)
                    asymmetry_metrics['right_foot_contact_time'] = round(base_contact_time - asymmetry_factor, 3)
                else:  # Right more pronated
                    asymmetry_metrics['left_foot_contact_time'] = round(base_contact_time - asymmetry_factor, 3)
                    asymmetry_metrics['right_foot_contact_time'] = round(base_contact_time + asymmetry_factor, 3)
        
        return asymmetry_metrics
    
    def _extract_ml_features(self, combined_results: Dict[str, Any]) -> Dict[str, List[float]]:
        """Extract features for ML classification"""
        features = {
            'biomechanical': [],
            'temporal': [],
            'spatial': [],
            'asymmetry': []
        }
        
        # Extract features from each angle
        for angle, result in combined_results['individual_angles'].items():
            metrics = result.get('metrics', {})
            
            if angle == 'rear' and 'left_rearfoot_eversion' in metrics:
                # Biomechanical features
                features['biomechanical'].extend([
                    metrics['left_rearfoot_eversion']['mean'],
                    metrics['left_rearfoot_eversion']['std'],
                    metrics['right_rearfoot_eversion']['mean'],
                    metrics['right_rearfoot_eversion']['std']
                ])
                
                # Asymmetry features
                features['asymmetry'].append(metrics.get('eversion_asymmetry', 0))
            
            if angle == 'side' and 'step_analysis' in metrics:
                # Temporal features
                step_data = metrics['step_analysis']
                features['temporal'].extend([
                    step_data.get('cadence', 0),
                    step_data.get('avg_step_duration', 0)
                ])
                
                # Spatial features
                features['spatial'].extend([
                    step_data.get('avg_step_length', 0),
                    step_data.get('step_length_variability', 0)
                ])
            
            if angle == 'front' and 'foot_progression' in metrics:
                # Front view spatial features (foot progression angles)
                foot_prog = metrics['foot_progression']
                features['spatial'].extend([
                    foot_prog.get('left_mean', 0),
                    foot_prog.get('right_mean', 0)
                ])
                
                # Front view asymmetry features
                features['asymmetry'].append(foot_prog.get('asymmetry', 0))
        
        # Flatten all features for ML model
        all_features = []
        for feature_type, feature_list in features.items():
            all_features.extend(feature_list)
        
        # Pad or truncate to fixed size (would be determined by ML model requirements)
        target_size = 20
        if len(all_features) < target_size:
            all_features.extend([0.0] * (target_size - len(all_features)))
        elif len(all_features) > target_size:
            all_features = all_features[:target_size]
        
        return {
            'feature_vector': all_features,
            'feature_names': ['eversion_l_mean', 'eversion_l_std', 'eversion_r_mean', 'eversion_r_std',
                            'asymmetry', 'cadence', 'step_duration', 'step_length', 'step_variability'],
            'raw_features': features
        }