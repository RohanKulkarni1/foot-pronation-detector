import React, { useState, useCallback } from 'react';
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
  Videocam,
  Upload,
  Analytics,
  CameraAlt,
  CheckCircle,
  RotateLeft,
  RotateRight,
  Visibility
} from '@mui/icons-material';
import axios from 'axios';
import config from '../config';
import CameraRecorder from '../components/CameraRecorder';

const MultiAngleCapture = () => {
  const navigate = useNavigate();
  const [activeStep, setActiveStep] = useState(0);
  const [sessionId, setSessionId] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showInstructions, setShowInstructions] = useState(false);
  const [selectedAngle, setSelectedAngle] = useState('rear');
  const [showCameraRecorder, setShowCameraRecorder] = useState(false);
  const [recordingAngle, setRecordingAngle] = useState('rear');

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
        'Walk 20-30 feet (6-9 meters) at normal pace',
        'Record 10-15 seconds for 8-10 complete gait cycles',
        'Include 2-3 steps before/after for acceleration/deceleration'
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
        'Walk 20-30 feet (6-9 meters) at normal pace',
        'Record 10-15 seconds for 8-10 complete gait cycles',
        'Maintain consistent walking speed throughout'
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
        'Walk 20-30 feet (6-9 meters) at normal pace',
        'Record 10-15 seconds for 8-10 complete gait cycles',
        'Capture symmetrical foot placement and stride width'
      ]
    }
  ];

  const handleAngleFileUpload = async (event, angle) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append('files', file);
      formData.append('angle', angle); // Specify the angle
      
      if (sessionId) {
        formData.append('session_id', sessionId);
      }

      const response = await axios.post(config.getApiUrl('/upload'), formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setSessionId(response.data.session_id);
      
      // Update uploaded files with angle information
      const newFiles = response.data.uploaded_files.map(uploadedFile => ({
        ...uploadedFile,
        angle: angle
      }));
      
      setUploadedFiles(prev => {
        // Remove any existing file for this angle
        const filtered = prev.filter(f => f.angle !== angle);
        return [...filtered, ...newFiles];
      });
      
      setActiveStep(2);
    } catch (error) {
      console.error('Upload error:', error);
    } finally {
      setIsUploading(false);
    }
  };

  const handleRecordingComplete = async (file, angle) => {
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append('files', file);
      
      if (sessionId) {
        formData.append('session_id', sessionId);
      }

      const response = await axios.post(config.getApiUrl('/upload'), formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setSessionId(response.data.session_id);
      setUploadedFiles(prev => [...prev, ...response.data.uploaded_files]);
      setActiveStep(2);
      setShowCameraRecorder(false);
    } catch (error) {
      console.error('Upload error:', error);
    } finally {
      setIsUploading(false);
    }
  };

  const openCameraRecorder = (angle) => {
    setRecordingAngle(angle);
    setShowCameraRecorder(true);
  };

  const analyzeGait = async () => {
    if (!sessionId || uploadedFiles.length === 0) {
      console.error('No session ID or files uploaded');
      alert('Please upload at least one video before analyzing');
      return;
    }

    setIsAnalyzing(true);
    setActiveStep(3);
    
    try {
      console.log('Starting analysis for session:', sessionId);
      const response = await axios.post(config.getApiUrl(`/analyze/${sessionId}`));
      console.log('Analysis complete:', response.data);
      
      // Navigate to results page with analysis data
      navigate(`/analysis/${sessionId}`);
    } catch (error) {
      console.error('Analysis error:', error);
      alert(`Analysis failed: ${error.response?.data?.detail || error.message}`);
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
        {/* File Upload by Angle */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Upload Videos by Angle
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Upload one video for each camera angle below
              </Typography>
              
              {cameraAngles.map((angle) => (
                <Box key={angle.key} sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    {angle.icon}
                    <Typography variant="subtitle1" sx={{ ml: 1, fontWeight: 500 }}>
                      {angle.label}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ ml: 2 }}>
                      - {angle.description}
                    </Typography>
                  </Box>
                  
                  <input
                    type="file"
                    accept="video/*"
                    id={`file-${angle.key}`}
                    style={{ display: 'none' }}
                    onChange={(e) => handleAngleFileUpload(e, angle.key)}
                  />
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Button
                      variant="outlined"
                      component="label"
                      htmlFor={`file-${angle.key}`}
                      startIcon={<Upload />}
                      disabled={isUploading}
                      size="small"
                    >
                      Choose {angle.label} Video
                    </Button>
                    
                    {uploadedFiles.find(f => f.angle === angle.key) && (
                      <Chip
                        label="✓ Uploaded"
                        color="success"
                        size="small"
                        variant="outlined"
                      />
                    )}
                  </Box>
                </Box>
              ))}
              
              {isUploading && (
                <Box sx={{ mt: 2 }}>
                  <LinearProgress />
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Uploading video...
                  </Typography>
                </Box>
              )}
              
              <Alert severity="info" sx={{ mt: 2 }}>
                <Typography variant="body2">
                  <strong>Tip:</strong> You can upload 1, 2, or all 3 angles. More angles = better analysis accuracy.
                </Typography>
              </Alert>
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

                {/* Camera Recorder or Preview */}
                {showCameraRecorder ? (
                  <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                      <Typography variant="subtitle1">
                        Recording {cameraAngles.find(a => a.key === recordingAngle)?.label}
                      </Typography>
                      <Button 
                        size="small" 
                        onClick={() => setShowCameraRecorder(false)}
                        color="secondary"
                      >
                        Close Camera
                      </Button>
                    </Box>
                    <CameraRecorder 
                      angle={recordingAngle}
                      onRecordingComplete={handleRecordingComplete}
                    />
                  </Box>
                ) : (
                  <>
                    <Paper sx={{ p: 2, backgroundColor: 'grey.100', mb: 2 }}>
                      <CameraAlt sx={{ fontSize: 64, color: 'grey.400' }} />
                      <Typography variant="body2" color="text.secondary">
                        Select angle and click to start recording
                      </Typography>
                    </Paper>
                    <Button
                      variant="contained"
                      color="primary"
                      onClick={() => openCameraRecorder(selectedAngle)}
                      startIcon={<Videocam />}
                      disabled={isUploading}
                    >
                      Open Camera for {cameraAngles.find(a => a.key === selectedAngle)?.label}
                    </Button>
                  </>
                )}
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
                    <Videocam sx={{ mr: 2, color: 'primary.main' }} />
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