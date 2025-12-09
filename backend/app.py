from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid
from pathlib import Path
import uvicorn
import json
from typing import List, Optional

# Import our Phase 2 modules
from analysis.enhanced_gait_analyzer import EnhancedGaitAnalyzer
from ml_models.gait_classifier import GaitClassifier
from datasets.dataset_manager import DatasetManager

app = FastAPI(
    title="Foot Analysis Phase 2 API",
    description="Advanced Gait Analysis with ML Classification",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
analyzer = EnhancedGaitAnalyzer()
classifier = GaitClassifier()
dataset_manager = DatasetManager()

# Create directories
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Mount static files (changed path to avoid conflict with API endpoint)
app.mount("/static-results", StaticFiles(directory=str(RESULTS_DIR)), name="static_results")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0", "phase": "2"}

@app.get("/debug-results/{session_id}")
async def debug_results(session_id: str):
    """Debug version of get_results"""
    results_file = RESULTS_DIR / f"{session_id}_results.json"
    
    return {
        "session_id": session_id,
        "looking_for": str(results_file),
        "absolute_path": str(results_file.absolute()),
        "current_dir": os.getcwd(),
        "results_dir": str(RESULTS_DIR),
        "results_dir_exists": RESULTS_DIR.exists(),
        "file_exists": results_file.exists(),
        "files_in_results": [f.name for f in RESULTS_DIR.glob("*") if f.is_file()][:10]
    }

@app.post("/upload")
async def upload_videos(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = None
):
    """Upload multiple videos for multi-angle analysis"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    uploaded_files = []
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    
    for file in files:
        # More flexible video type checking
        valid_extensions = ('.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI')
        if not (file.content_type.startswith("video/") or 
                file.filename.lower().endswith(valid_extensions)):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a video. Type: {file.content_type}")
        
        file_path = session_dir / f"{file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        uploaded_files.append({
            "filename": file.filename,
            "path": str(file_path),
            "angle": "rear"  # Default, frontend should specify
        })
    
    return {
        "session_id": session_id,
        "uploaded_files": uploaded_files,
        "message": f"Uploaded {len(files)} video(s)"
    }

@app.post("/analyze/{session_id}")
async def analyze_gait(session_id: str):
    """Enhanced multi-angle gait analysis"""
    session_dir = UPLOAD_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get all video files in session (case-insensitive)
    video_files = []
    for pattern in ["*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI"]:
        video_files.extend(session_dir.glob(pattern))
    
    if not video_files:
        # Log what files exist for debugging
        all_files = list(session_dir.glob("*"))
        file_list = [f.name for f in all_files]
        raise HTTPException(status_code=404, detail=f"No video files found. Files in session: {file_list}")
    
    try:
        # Enhanced analysis with ML classification
        results = await analyzer.analyze_multi_angle(video_files, session_id)
        
        # ML Classification
        ml_classification = classifier.classify_gait(results['features'])
        
        # Extract the proper classification data from the response
        if 'ensemble_prediction' in ml_classification:
            # Full classification response with clinical data
            results['ml_classification'] = {
                'primary_classification': ml_classification['ensemble_prediction'].get('class', 'neutral'),
                'confidence': ml_classification['ensemble_prediction'].get('confidence', 0.0),
                'model_used': ml_classification.get('method_used', 'clinical_fallback'),
                'confidence_scores': ml_classification['ensemble_prediction'].get('class_probabilities', {}),
                'clinical_prediction': ml_classification.get('clinical_prediction', {}),
                'clinical_validation': ml_classification.get('clinical_validation', {}),
                'individual_predictions': ml_classification.get('individual_predictions', {}),
                'feature_importance': ml_classification.get('feature_importance', {})
            }
            
            # Also add clinical classification as separate field for backward compatibility
            if ml_classification.get('clinical_prediction'):
                results['clinical_classification'] = ml_classification['clinical_prediction']
        else:
            # Simplified response (fallback mode)
            results['ml_classification'] = ml_classification
        
        # Save results
        results_file = RESULTS_DIR / f"{session_id}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        return JSONResponse(content=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{session_id}")
async def get_results(session_id: str):
    """Get analysis results"""
    results_file = RESULTS_DIR / f"{session_id}_results.json"
    if not results_file.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    
    with open(results_file, "r") as f:
        results = json.load(f)
    return JSONResponse(content=results)

@app.get("/datasets")
async def list_datasets():
    """List available training datasets"""
    return dataset_manager.list_datasets()

@app.post("/datasets/upload")
async def upload_training_data(
    file: UploadFile = File(...),
    label: str = "unknown"
):
    """Upload labeled training data"""
    return await dataset_manager.add_training_sample(file, label)

@app.get("/model/metrics")
async def get_model_metrics():
    """Get ML model performance metrics"""
    return classifier.get_performance_metrics()

@app.post("/model/retrain")
async def retrain_model():
    """Retrain the ML classifier with new data"""
    try:
        metrics = await classifier.retrain()
        return {"message": "Model retrained successfully", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)