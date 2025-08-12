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

# Mount static files
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0", "phase": "2"}

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
        if not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a video")
        
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
    
    # Get all video files in session
    video_files = list(session_dir.glob("*.mp4")) + list(session_dir.glob("*.mov"))
    if not video_files:
        raise HTTPException(status_code=404, detail="No video files found")
    
    try:
        # Enhanced analysis with ML classification
        results = await analyzer.analyze_multi_angle(video_files, session_id)
        
        # ML Classification
        ml_classification = classifier.classify_gait(results['features'])
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