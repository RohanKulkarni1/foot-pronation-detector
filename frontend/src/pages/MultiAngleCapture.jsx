import React, { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Paper,
  LinearProgress,
  Chip,
  Alert,
  Stepper,
  Step,
  StepLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import {
  VideoCamera,
  Upload,
  Analytics,
  CameraAlt,
  CheckCircle,
  Warning,
  Info,
  RotateLeft,
  RotateRight,
  Visibility
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import Webcam from 'react-webcam';
import axios from 'axios';

const MultiAngleCapture = () => {
  const navigate = useNavigate();
  const [activeStep, setActiveStep] = useState(0);
  const [sessionId, setSessionId] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showInstructions, setShowInstructions] = useState(false);
  const [selectedAngle, setSelectedAngle] = useState('rear');
  const [recordingStates, setRecordingStates] = useState({
    rear: false,
    side: false,
    front: false
  });

  const webcamRefs = {
    rear: useRef(null),
    side: useRef(null),
    front: useRef(null)
  };

  const steps = ['Setup', 'Record/Upload', 'Process', 'Analyze'];

  const cameraAngles = [
    {
      key: 'rear',
      label: 'Rear View',
      description: 'For pronation/supination analysis',
      icon: <RotateLeft />,
      instructions: [
        'Position camera directly behind subject',
        'Ensure both feet and lower legs are visible',
        'Subject should walk away from camera',
        'Capture 10-15 steps for best results'
      ]
    },
    {
      key: 'side',
      label: 'Side View', 
      description: 'For step length and cadence analysis',
      icon: <Visibility />,
      instructions: [
        'Position camera perpendicular to walking path',
        'Capture full body profile if possible',
        'Ensure clear view of knee and ankle movement',
        'Record complete gait cycles'
      ]
    },
    {
      key: 'front',
      label: 'Front View',
      description: 'For foot progression angle',
      icon: <RotateRight />,
      instructions: [
        'Position camera directly in front of subject',
        'Focus on feet and lower leg alignment',
        'Subject should walk toward camera',
        'Capture symmetrical foot placement'
      ]
    }
  ];

  const onDrop = useCallback(async (acceptedFiles) => {
    setIsUploading(true);
    try {
      const formData = new FormData();
      acceptedFiles.forEach(file => {
        // Try to detect angle from filename or let user specify
        formData.append('files', file);
      });
      
      if (sessionId) {
        formData.append('session_id', sessionId);
      }

      const response = await axios.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setSessionId(response.data.session_id);
      setUploadedFiles(prev => [...prev, ...response.data.uploaded_files]);
      setActiveStep(2);
    } catch (error) {
      console.error('Upload error:', error);
    } finally {
      setIsUploading(false);
    }
  }, [sessionId]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov', '.avi', '.webm']
    },
    multiple: true
  });

  const startRecording = (angle) => {
    setRecordingStates(prev => ({
      ...prev,
      [angle]: true
    }));
    
    // In a real implementation, you would start MediaRecorder here
    // For demo purposes, we'll simulate recording
    setTimeout(() => {
      stopRecording(angle);
    }, 5000);
  };

  const stopRecording = (angle) => {
    setRecordingStates(prev => ({
      ...prev,
      [angle]: false
    }));
    
    // Simulate adding recorded file
    const newFile = {
      filename: `${angle}_view_recording.webm`,
      angle: angle,
      timestamp: new Date().toISOString()
    };
    
    setUploadedFiles(prev => [...prev, newFile]);
  };

  const analyzeGait = async () => {
    if (!sessionId || uploadedFiles.length === 0) return;

    setIsAnalyzing(true);
    setActiveStep(3);
    
    try {
      const response = await axios.post(`/analyze/${sessionId}`);
      // Navigate to results page with analysis data
      navigate(`/analysis/${sessionId}`);
    } catch (error) {
      console.error('Analysis error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const InstructionsDialog = () => (
    <Dialog open={showInstructions} onClose={() => setShowInstructions(false)} maxWidth="md">
      <DialogTitle>Recording Instructions</DialogTitle>
      <DialogContent>
        <Grid container spacing={3}>
          {cameraAngles.map((angle) => (
            <Grid item xs={12} md={4} key={angle.key}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Box sx={{ color: 'primary.main', mb: 1 }}>
                  {angle.icon}
                </Box>
                <Typography variant="h6" gutterBottom>
                  {angle.label}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {angle.description}
                </Typography>
                <List dense>
                  {angle.instructions.map((instruction, index) => (
                    <ListItem key={index} sx={{ px: 0 }}>
                      <ListItemIcon sx={{ minWidth: 36 }}>
                        <CheckCircle color="success" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText 
                        primary={instruction} 
                        primaryTypographyProps={{ fontSize: '0.875rem' }}
                      />
                    </ListItem>
                  ))}
                </List>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setShowInstructions(false)}>Close</Button>
      </DialogActions>
    </Dialog>
  );

  return (
    <Box>
      {/* Header */}
      <Typography variant="h4" gutterBottom sx={{ fontWeight: 600 }}>
        Multi-Angle Gait Capture
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Capture or upload videos from multiple angles for comprehensive 3D gait analysis
      </Typography>

      {/* Progress Stepper */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Stepper activeStep={activeStep} alternativeLabel>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
      </Paper>

      {/* Instructions Alert */}
      <Alert 
        severity="info" 
        action={
          <Button color="inherit" size="small" onClick={() => setShowInstructions(true)}>
            View Instructions
          </Button>
        }
        sx={{ mb: 3 }}
      >
        For best results, capture videos from rear, side, and front angles. Click "View Instructions" for detailed setup guidance.
      </Alert>

      {/* Upload/Record Section */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* File Upload */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Upload Video Files
              </Typography>
              <Paper
                {...getRootProps()}
                sx={{
                  p: 3,
                  textAlign: 'center',
                  border: '2px dashed',
                  borderColor: isDragActive ? 'primary.main' : 'grey.300',
                  backgroundColor: isDragActive ? 'primary.light' : 'grey.50',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                <input {...getInputProps()} />
                <Upload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  {isDragActive ? 'Drop videos here' : 'Drag & drop videos'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Support: MP4, MOV, AVI, WebM
                </Typography>
                <Button variant="contained" sx={{ mt: 2 }}>
                  Choose Files
                </Button>
              </Paper>
              {isUploading && <LinearProgress sx={{ mt: 2 }} />}
            </CardContent>
          </Card>
        </Grid>

        {/* Live Recording */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Live Recording
              </Typography>
              <Box sx={{ textAlign: 'center', mb: 2 }}>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Select camera angle and record directly
                </Typography>
                
                {/* Angle Selection */}
                <Box sx={{ mb: 2 }}>
                  {cameraAngles.map((angle) => (
                    <Chip
                      key={angle.key}
                      label={angle.label}
                      onClick={() => setSelectedAngle(angle.key)}
                      color={selectedAngle === angle.key ? 'primary' : 'default'}
                      sx={{ mr: 1, mb: 1 }}
                    />
                  ))}
                </Box>

                {/* Camera Preview (simplified) */}
                <Paper sx={{ p: 2, backgroundColor: 'grey.100', mb: 2 }}>
                  <CameraAlt sx={{ fontSize: 64, color: 'grey.400' }} />
                  <Typography variant="body2" color="text.secondary">
                    Camera Preview - {cameraAngles.find(a => a.key === selectedAngle)?.label}
                  </Typography>
                </Paper>

                {/* Recording Controls */}
                <Button
                  variant={recordingStates[selectedAngle] ? "contained" : "outlined"}
                  color={recordingStates[selectedAngle] ? "error" : "primary"}
                  onClick={() => recordingStates[selectedAngle] ? 
                    stopRecording(selectedAngle) : 
                    startRecording(selectedAngle)
                  }
                  startIcon={<VideoCamera />}
                  disabled={isUploading}
                >
                  {recordingStates[selectedAngle] ? 'Stop Recording' : 'Start Recording'}
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Uploaded Files */}
      {uploadedFiles.length > 0 && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Uploaded Files ({uploadedFiles.length})
            </Typography>
            <Grid container spacing={2}>
              {uploadedFiles.map((file, index) => (
                <Grid item xs={12} sm={6} md={4} key={index}>
                  <Paper sx={{ p: 2, display: 'flex', alignItems: 'center' }}>
                    <VideoCamera sx={{ mr: 2, color: 'primary.main' }} />
                    <Box sx={{ flexGrow: 1 }}>
                      <Typography variant="body2" noWrap>
                        {file.filename}
                      </Typography>
                      <Chip 
                        label={file.angle || 'Unknown angle'} 
                        size="small" 
                        color="primary" 
                        variant="outlined" 
                      />
                    </Box>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Analysis Button */}
      {uploadedFiles.length > 0 && (
        <Box sx={{ textAlign: 'center' }}>
          <Button
            variant="contained"
            size="large"
            onClick={analyzeGait}
            disabled={isAnalyzing}
            startIcon={<Analytics />}
            sx={{ minWidth: 200 }}
          >
            {isAnalyzing ? 'Analyzing...' : 'Start Advanced Analysis'}
          </Button>
          {isAnalyzing && <LinearProgress sx={{ mt: 2 }} />}
        </Box>
      )}

      <InstructionsDialog />
    </Box>
  );
};

export default MultiAngleCapture;