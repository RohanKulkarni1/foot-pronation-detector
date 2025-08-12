import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import shutil
import uuid
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetManager:
    """
    Manages training datasets for Phase 2 ML models
    Handles data loading, labeling, validation, and organization
    """
    
    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = Path(data_dir)
        self.training_dir = self.data_dir / "training"
        self.clinical_dir = self.data_dir / "clinical"
        self.samples_dir = self.data_dir / "samples"
        
        # Create directories
        for dir_path in [self.training_dir, self.clinical_dir, self.samples_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Metadata files
        self.training_metadata = self.training_dir / "metadata.json"
        self.clinical_metadata = self.clinical_dir / "metadata.json"
        
        # Initialize metadata if not exists
        self._initialize_metadata()
    
    def _initialize_metadata(self):
        """Initialize metadata files"""
        if not self.training_metadata.exists():
            initial_metadata = {
                "created": datetime.now().isoformat(),
                "version": "2.0.0",
                "samples": [],
                "labels": {
                    "neutral": {"count": 0, "description": "Normal gait pattern"},
                    "mild_pronation": {"count": 0, "description": "Slight overpronation"},
                    "moderate_pronation": {"count": 0, "description": "Moderate overpronation"},
                    "severe_pronation": {"count": 0, "description": "Severe overpronation"},
                    "supination": {"count": 0, "description": "Underpronation/supination"}
                },
                "data_sources": []
            }
            
            with open(self.training_metadata, 'w') as f:
                json.dump(initial_metadata, f, indent=2)
    
    def list_datasets(self) -> Dict[str, Any]:
        """List all available datasets"""
        training_metadata = self._load_metadata(self.training_metadata)
        clinical_metadata = self._load_metadata(self.clinical_metadata)
        
        # Get sample files
        sample_files = list(self.samples_dir.glob("*.mp4")) + list(self.samples_dir.glob("*.mov"))
        
        return {
            "training_dataset": {
                "total_samples": len(training_metadata.get("samples", [])),
                "label_distribution": training_metadata.get("labels", {}),
                "last_updated": training_metadata.get("last_updated", "Never"),
                "data_sources": training_metadata.get("data_sources", [])
            },
            "clinical_dataset": {
                "total_samples": len(clinical_metadata.get("samples", [])),
                "last_updated": clinical_metadata.get("last_updated", "Never")
            },
            "sample_videos": {
                "count": len(sample_files),
                "files": [f.name for f in sample_files[:10]]  # Show first 10
            }
        }
    
    async def add_training_sample(self, file, label: str) -> Dict[str, Any]:
        """Add a new training sample with label"""
        try:
            # Generate unique sample ID
            sample_id = str(uuid.uuid4())
            
            # Save file
            file_extension = Path(file.filename).suffix
            sample_path = self.training_dir / f"{sample_id}{file_extension}"
            
            with open(sample_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Update metadata
            metadata = self._load_metadata(self.training_metadata)
            
            sample_info = {
                "id": sample_id,
                "filename": file.filename,
                "stored_path": str(sample_path),
                "label": label,
                "added_date": datetime.now().isoformat(),
                "file_size": len(content),
                "status": "unlabeled" if label == "unknown" else "labeled"
            }
            
            metadata["samples"].append(sample_info)
            metadata["labels"][label]["count"] += 1
            metadata["last_updated"] = datetime.now().isoformat()
            
            # Save updated metadata
            with open(self.training_metadata, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Added training sample: {sample_id} with label: {label}")
            
            return {
                "success": True,
                "sample_id": sample_id,
                "label": label,
                "message": f"Successfully added training sample"
            }
            
        except Exception as e:
            logger.error(f"Error adding training sample: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_training_data(self) -> List[Dict[str, Any]]:
        """Get all training data for ML model training"""
        metadata = self._load_metadata(self.training_metadata)
        
        training_data = []
        for sample in metadata.get("samples", []):
            if sample.get("status") == "labeled" and sample.get("features"):
                training_data.append({
                    "features": sample["features"],
                    "label": sample["label"],
                    "sample_id": sample["id"]
                })
        
        return training_data
    
    def update_sample_features(self, sample_id: str, features: List[float]) -> bool:
        """Update extracted features for a sample"""
        try:
            metadata = self._load_metadata(self.training_metadata)
            
            for sample in metadata["samples"]:
                if sample["id"] == sample_id:
                    sample["features"] = features
                    sample["features_extracted"] = datetime.now().isoformat()
                    break
            
            metadata["last_updated"] = datetime.now().isoformat()
            
            with open(self.training_metadata, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating features for {sample_id}: {str(e)}")
            return False
    
    def relabel_sample(self, sample_id: str, new_label: str) -> Dict[str, Any]:
        """Relabel an existing sample"""
        try:
            metadata = self._load_metadata(self.training_metadata)
            
            sample_found = False
            old_label = None
            
            for sample in metadata["samples"]:
                if sample["id"] == sample_id:
                    old_label = sample.get("label", "unknown")
                    sample["label"] = new_label
                    sample["relabeled_date"] = datetime.now().isoformat()
                    sample["status"] = "labeled" if new_label != "unknown" else "unlabeled"
                    sample_found = True
                    break
            
            if not sample_found:
                return {"success": False, "error": "Sample not found"}
            
            # Update label counts
            if old_label and old_label in metadata["labels"]:
                metadata["labels"][old_label]["count"] = max(0, metadata["labels"][old_label]["count"] - 1)
            
            if new_label in metadata["labels"]:
                metadata["labels"][new_label]["count"] += 1
            
            metadata["last_updated"] = datetime.now().isoformat()
            
            with open(self.training_metadata, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                "success": True,
                "old_label": old_label,
                "new_label": new_label,
                "message": f"Successfully relabeled sample from {old_label} to {new_label}"
            }
            
        except Exception as e:
            logger.error(f"Error relabeling sample {sample_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def export_dataset(self, format: str = "json") -> Dict[str, Any]:
        """Export dataset in various formats"""
        try:
            training_data = self.get_training_data()
            
            if format == "json":
                export_path = self.data_dir / f"dataset_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(export_path, 'w') as f:
                    json.dump(training_data, f, indent=2)
            
            elif format == "csv":
                if training_data:
                    df_data = []
                    for item in training_data:
                        row = {"label": item["label"], "sample_id": item["sample_id"]}
                        for i, feature in enumerate(item["features"]):
                            row[f"feature_{i}"] = feature
                        df_data.append(row)
                    
                    df = pd.DataFrame(df_data)
                    export_path = self.data_dir / f"dataset_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    df.to_csv(export_path, index=False)
                else:
                    return {"success": False, "error": "No training data available"}
            
            else:
                return {"success": False, "error": f"Unsupported format: {format}"}
            
            return {
                "success": True,
                "export_path": str(export_path),
                "format": format,
                "samples_exported": len(training_data)
            }
            
        except Exception as e:
            logger.error(f"Error exporting dataset: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def import_external_dataset(self, dataset_path: str, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Import external gait dataset"""
        try:
            dataset_path = Path(dataset_path)
            
            if not dataset_path.exists():
                return {"success": False, "error": "Dataset path does not exist"}
            
            # Create import record
            import_id = str(uuid.uuid4())
            import_record = {
                "import_id": import_id,
                "source_path": str(dataset_path),
                "dataset_info": dataset_info,
                "import_date": datetime.now().isoformat(),
                "status": "imported"
            }
            
            # Update metadata
            metadata = self._load_metadata(self.training_metadata)
            metadata["data_sources"].append(import_record)
            metadata["last_updated"] = datetime.now().isoformat()
            
            with open(self.training_metadata, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                "success": True,
                "import_id": import_id,
                "message": "External dataset imported successfully"
            }
            
        except Exception as e:
            logger.error(f"Error importing external dataset: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Generate data quality report"""
        metadata = self._load_metadata(self.training_metadata)
        samples = metadata.get("samples", [])
        
        total_samples = len(samples)
        labeled_samples = len([s for s in samples if s.get("status") == "labeled"])
        
        # Label distribution
        label_dist = {}
        for sample in samples:
            label = sample.get("label", "unknown")
            label_dist[label] = label_dist.get(label, 0) + 1
        
        # Quality metrics
        quality_metrics = {
            "total_samples": total_samples,
            "labeled_samples": labeled_samples,
            "unlabeled_samples": total_samples - labeled_samples,
            "labeling_completeness": labeled_samples / total_samples if total_samples > 0 else 0,
            "label_distribution": label_dist,
            "class_balance": self._calculate_class_balance(label_dist),
            "missing_features": len([s for s in samples if not s.get("features")]),
            "data_sources": len(metadata.get("data_sources", []))
        }
        
        # Recommendations
        recommendations = self._generate_recommendations(quality_metrics)
        
        return {
            "quality_metrics": quality_metrics,
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat()
        }
    
    def _calculate_class_balance(self, label_dist: Dict[str, int]) -> Dict[str, float]:
        """Calculate class balance metrics"""
        total = sum(label_dist.values())
        if total == 0:
            return {}
        
        balance = {}
        for label, count in label_dist.items():
            balance[label] = count / total
        
        return balance
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations for data quality improvement"""
        recommendations = []
        
        if metrics["labeling_completeness"] < 0.8:
            recommendations.append("Consider labeling more samples to improve model training quality")
        
        if metrics["total_samples"] < 100:
            recommendations.append("Collect more training samples for better model performance")
        
        # Check class balance
        balance = metrics.get("class_balance", {})
        if balance:
            min_ratio = min(balance.values())
            if min_ratio < 0.1:
                recommendations.append("Some classes are underrepresented. Consider collecting more samples for minority classes")
        
        if metrics["missing_features"] > 0:
            recommendations.append(f"{metrics['missing_features']} samples are missing extracted features")
        
        return recommendations
    
    def _load_metadata(self, metadata_path: Path) -> Dict[str, Any]:
        """Load metadata file"""
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {"samples": [], "labels": {}}
    
    def cleanup_orphaned_files(self) -> Dict[str, Any]:
        """Remove files that are not referenced in metadata"""
        try:
            metadata = self._load_metadata(self.training_metadata)
            referenced_files = {sample.get("stored_path") for sample in metadata.get("samples", [])}
            
            # Find all files in training directory
            all_files = set()
            for ext in ["*.mp4", "*.mov", "*.avi", "*.json"]:
                all_files.update(str(f) for f in self.training_dir.glob(ext))
            
            # Remove metadata file from cleanup candidates
            all_files.discard(str(self.training_metadata))
            
            # Find orphaned files
            orphaned_files = all_files - referenced_files
            
            # Remove orphaned files
            removed_count = 0
            for file_path in orphaned_files:
                try:
                    Path(file_path).unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Could not remove {file_path}: {e}")
            
            return {
                "success": True,
                "removed_files": removed_count,
                "orphaned_files": list(orphaned_files)
            }
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            return {"success": False, "error": str(e)}