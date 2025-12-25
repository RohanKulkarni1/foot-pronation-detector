import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Alert,
  Button,
  Paper,
  Chip,
  Divider,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Analytics,
  Speed,
  Timeline,
  Assessment,
  Download,
  ArrowBack,
  CheckCircle,
  Warning,
  Info,
  LocalHospital,
  Psychology,
  Videocam,
  PlayArrow
} from '@mui/icons-material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  RadialLinearScale,
  ArcElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line, Bar, Radar, Doughnut } from 'react-chartjs-2';
import axios from 'axios';
import config from '../config';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  RadialLinearScale,
  ArcElement,
  Title,
  ChartTooltip,
  Legend,
  Filler
);

const AdvancedAnalysis = () => {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [visualizations, setVisualizations] = useState([]);
  const [creatingVisualizations, setCreatingVisualizations] = useState(false);
  const [showVisualizationTab, setShowVisualizationTab] = useState(false);

  useEffect(() => {
    if (sessionId) {
      fetchResults();
    } else {
      // If no sessionId, check for latest analysis
      setLoading(false);
    }
  }, [sessionId]);

  const fetchResults = async () => {
    console.log('🚀 FETCHING RESULTS FOR SESSION:', sessionId);
    try {
      setLoading(true);
      const url = config.getApiUrl(`/results/${sessionId}?_t=${Date.now()}`);
      console.log('🔗 FETCH URL:', url);
      const response = await axios.get(url, {
        headers: {
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache'
        }
      });
      console.log('✅ Raw response data:', response.data);
      console.log('🤖 ML Classification:', response.data.ml_classification);
      console.log('📊 Confidence Scores:', response.data.ml_classification?.confidence_scores);
      setResults(response.data);
      
      // Check if visualizations exist - temporarily disabled since endpoint doesn't exist
      // await checkExistingVisualizations();
    } catch (err) {
      setError('Failed to load analysis results. Please try again.');
      console.error('Error fetching results:', err);
    } finally {
      setLoading(false);
    }
  };

  const checkExistingVisualizations = async () => {
    try {
      const url = config.getApiUrl(`/visualizations/${sessionId}`);
      const response = await axios.get(url);
      setVisualizations(response.data.visualizations);
      setShowVisualizationTab(true);
    } catch (err) {
      // No existing visualizations found - this is normal
      console.log('No existing visualizations found');
    }
  };

  const createVisualizations = async () => {
    try {
      setCreatingVisualizations(true);
      const url = config.getApiUrl(`/visualize/${sessionId}`);
      const response = await axios.post(url);
      setVisualizations(response.data.visualizations);
      setShowVisualizationTab(true);
      setActiveTab(4); // Switch to visualization tab
    } catch (err) {
      console.error('Error creating visualizations:', err);
      alert('Failed to create visualizations. Please try again.');
    } finally {
      setCreatingVisualizations(false);
    }
  };

  const getClassificationColor = (classification) => {
    const colors = {
      'Neutral': 'success',
      'Mild Pronation': 'info',
      'Mild Overpronation': 'info',
      'Moderate Pronation': 'warning',
      'Moderate Overpronation': 'warning',
      'Severe Pronation': 'error',
      'Severe Overpronation': 'error',
      'Supination': 'secondary'
    };
    return colors[classification] || 'default';
  };

  const getRecommendationIcon = (type) => {
    const icons = {
      'footwear': <LocalHospital />,
      'exercise': <Speed />,
      'medical': <Psychology />
    };
    return icons[type] || <Info />;
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 8 }}>
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Analyzing gait patterns...
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          This may take a few moments
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
        <Button variant="contained" onClick={() => navigate('/multi-angle')}>
          Back to Capture
        </Button>
      </Box>
    );
  }

  if (!results) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Advanced Analysis
        </Typography>
        <Alert severity="info" sx={{ mb: 2 }}>
          No analysis results available. Please upload videos first.
        </Alert>
        <Button 
          variant="contained" 
          startIcon={<Analytics />}
          onClick={() => navigate('/multi-angle')}
        >
          Start New Analysis
        </Button>
      </Box>
    );
  }

  // Prepare chart data
  const stepMetricsData = {
    labels: results.step_metrics?.map((_, i) => `Step ${i + 1}`) || [],
    datasets: [
      {
        label: 'Step Duration (s)',
        data: results.step_metrics?.map(m => m.duration) || [],
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1
      }
    ]
  };

  const asymmetryData = {
    labels: ['Left Foot', 'Right Foot'],
    datasets: [
      {
        label: 'Eversion Angle (°)',
        data: [
          Math.abs(results.biomechanical_profile?.foot_mechanics?.left_foot?.eversion_angle || 0),
          Math.abs(results.biomechanical_profile?.foot_mechanics?.right_foot?.eversion_angle || 0)
        ],
        backgroundColor: ['rgba(54, 162, 235, 0.5)', 'rgba(255, 99, 132, 0.5)'],
        borderColor: ['rgb(54, 162, 235)', 'rgb(255, 99, 132)'],
        borderWidth: 1
      }
    ]
  };

  const biomechanicalRadarData = {
    labels: [
      'Cadence',
      'Step Length',
      'Symmetry',
      'Efficiency',
      'Stability',
      'Balance'
    ],
    datasets: [
      {
        label: 'Current Analysis',
        data: [
          results.biomechanical_metrics?.cadence_score || 0,
          results.biomechanical_metrics?.step_length_score || 0,
          results.biomechanical_metrics?.symmetry_score || 0,
          results.biomechanical_metrics?.efficiency_score || 0,
          results.biomechanical_metrics?.stability_score || 0,
          results.biomechanical_metrics?.balance_score || 0
        ],
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderColor: 'rgb(255, 99, 132)',
        pointBackgroundColor: 'rgb(255, 99, 132)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgb(255, 99, 132)'
      },
      {
        label: 'Normal Range',
        data: [85, 85, 90, 85, 90, 85],
        backgroundColor: 'rgba(54, 162, 235, 0.1)',
        borderColor: 'rgb(54, 162, 235)',
        pointBackgroundColor: 'rgb(54, 162, 235)',
        pointBorderColor: '#fff',
        borderDash: [5, 5]
      }
    ]
  };

  const mlConfidenceData = {
    labels: Object.keys(results.ml_classification?.confidence_scores || {}).map(label => 
      label.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    ),
    datasets: [
      {
        data: Object.values(results.ml_classification?.confidence_scores || {}),
        backgroundColor: [
          'rgba(75, 192, 192, 0.8)',
          'rgba(54, 162, 235, 0.8)',
          'rgba(255, 206, 86, 0.8)',
          'rgba(255, 99, 132, 0.8)',
          'rgba(153, 102, 255, 0.8)'
        ],
        borderColor: [
          'rgba(75, 192, 192, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(255, 99, 132, 1)',
          'rgba(153, 102, 255, 1)'
        ],
        borderWidth: 2
      }
    ]
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <IconButton onClick={() => navigate('/multi-angle')} sx={{ mr: 2 }}>
          <ArrowBack />
        </IconButton>
        <Box sx={{ flexGrow: 1 }}>
          <Typography variant="h4" gutterBottom>
            Gait Analysis Results
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2 }}>
          {!showVisualizationTab && (
            <Button
              variant="contained"
              startIcon={<Videocam />}
              onClick={createVisualizations}
              disabled={creatingVisualizations}
              color="secondary"
            >
              {creatingVisualizations ? 'Creating...' : 'Show Video Analysis'}
            </Button>
          )}
          <Button
            variant="outlined"
            startIcon={<Download />}
            onClick={() => window.print()}
          >
            Export Report
          </Button>
        </Box>
      </Box>

      {/* ML Classification Summary */}
      <Card sx={{ mb: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
        <CardContent>
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} md={6}>
              <Typography variant="h6" sx={{ color: 'white', mb: 1 }}>
                ML Classification Result
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Chip
                  label={results.ml_classification?.primary_classification || 'Analyzing...'}
                  color={getClassificationColor(results.ml_classification?.primary_classification)}
                  size="large"
                  sx={{ fontSize: '1.1rem', py: 3 }}
                />
                <Typography variant="h4" sx={{ color: 'white' }}>
                  {(results.ml_classification?.confidence * 100 || 0).toFixed(1)}%
                </Typography>
                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.8)' }}>
                  Confidence
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.9)', mb: 1 }}>
                Model Used: {results.ml_classification?.model_used || 'Ensemble'}
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Tabs for Different Analysis Sections */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)}>
          <Tab label="Biomechanics" icon={<Timeline />} iconPosition="start" />
          <Tab label="ML Analysis" icon={<Psychology />} iconPosition="start" />
          <Tab label="Step Metrics" icon={<Speed />} iconPosition="start" />
          <Tab label="Recommendations" icon={<Assessment />} iconPosition="start" />
          {showVisualizationTab && (
            <Tab label="Video Analysis" icon={<Videocam />} iconPosition="start" />
          )}
        </Tabs>
      </Paper>

      {/* Tab Content */}
      {activeTab === 0 && (
        <Grid container spacing={3}>
          {/* Biomechanical Radar Chart */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Biomechanical Profile
                </Typography>
                <Box sx={{ height: 300 }}>
                  <Radar 
                    data={biomechanicalRadarData} 
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      scales: {
                        r: {
                          beginAtZero: true,
                          max: 100
                        }
                      }
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Asymmetry Analysis */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Left-Right Asymmetry
                </Typography>
                <Box sx={{ height: 300 }}>
                  <Bar 
                    data={asymmetryData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          display: false
                        }
                      }
                    }}
                  />
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                  Asymmetry Difference: {results.biomechanical_profile?.asymmetry_analysis?.degree_difference?.toFixed(2) || 'N/A'}° ({results.biomechanical_profile?.asymmetry_analysis?.level || 'Unknown'} level)
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Key Metrics */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Key Biomechanical Metrics
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'grey.50' }}>
                      <Typography variant="h4" color="primary">
                        {results.biomechanical_metrics?.cadence || 'N/A'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Cadence (steps/min)
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'grey.50' }}>
                      <Typography variant="h4" color="primary">
                        {results.biomechanical_metrics?.step_length?.toFixed(2) || 'N/A'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Avg Step Length (m)
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'grey.50' }}>
                      <Typography variant="h4" color="primary">
                        {results.biomechanical_metrics?.stance_time?.toFixed(2) || 'N/A'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Stance Time (s)
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'grey.50' }}>
                      <Typography variant="h4" color="primary">
                        {results.biomechanical_metrics?.swing_time?.toFixed(2) || 'N/A'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Swing Time (s)
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {activeTab === 1 && (
        <Grid container spacing={3}>
          {/* ML Confidence Scores */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Classification Confidence Scores
                </Typography>
                

                {Object.keys(results.ml_classification?.confidence_scores || {}).length > 0 ? (
                  <>
                    <Box sx={{ height: 300 }}>
                      <Doughnut 
                        data={mlConfidenceData}
                        options={{
                          responsive: true,
                          maintainAspectRatio: false,
                          plugins: {
                            legend: {
                              position: 'bottom'
                            },
                            tooltip: {
                              callbacks: {
                                label: function(context) {
                                  const label = context.label || '';
                                  const value = (context.parsed * 100).toFixed(1);
                                  return `${label}: ${value}%`;
                                }
                              }
                            }
                          }
                        }}
                      />
                    </Box>
                    
                    {/* Alternative display if chart fails */}
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Confidence Breakdown:
                      </Typography>
                      {Object.entries(results.ml_classification?.confidence_scores || {}).map(([key, value]) => (
                        <Typography key={key} variant="body2">
                          {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}: {(value * 100).toFixed(1)}%
                        </Typography>
                      ))}
                    </Box>
                  </>
                ) : (
                  <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Typography variant="body2" color="text.secondary" align="center">
                      {results.ml_classification ? 
                        'No confidence score breakdown available for this analysis' : 
                        'No ML classification data available'
                      }
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Feature Importance */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Key Features for Classification
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Feature</TableCell>
                        <TableCell align="right">Value</TableCell>
                        <TableCell align="right">Importance</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {results.features?.feature_vector?.slice(0, 8).map((value, index) => (
                        <TableRow key={index}>
                          <TableCell>
                            {results.features?.feature_names?.[index] || `Feature ${index + 1}`}
                          </TableCell>
                          <TableCell align="right">
                            {typeof value === 'number' ? value.toFixed(3) : 'N/A'}
                          </TableCell>
                          <TableCell align="right">
                            <LinearProgress 
                              variant="determinate" 
                              value={Math.abs(value) * 20} 
                              sx={{ minWidth: 50 }}
                            />
                          </TableCell>
                        </TableRow>
                      )) || (
                        <TableRow>
                          <TableCell colSpan={3}>No feature data available</TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>

          {/* Model Performance */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Model Performance Metrics
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h5" color="success.main">
                        {((results.ml_classification?.confidence || 0) * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body2">Confidence</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h5" color="info.main">
                        {results.ml_classification?.model_used || 'Clinical'}
                      </Typography>
                      <Typography variant="body2">Method</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h5" color="warning.main">
                        {Object.keys(results.ml_classification?.confidence_scores || {}).length || 0}
                      </Typography>
                      <Typography variant="body2">Classes</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h5" color="secondary.main">
                        {results.ml_classification?.clinical_validation?.is_valid ? 'Valid' : 'N/A'}
                      </Typography>
                      <Typography variant="body2">Clinical Valid</Typography>
                    </Paper>
                  </Grid>
                </Grid>
                
                {/* Show confidence breakdown if available */}
                {results.ml_classification?.confidence_scores && (
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Classification Confidence Breakdown
                    </Typography>
                    <Grid container spacing={2}>
                      {Object.entries(results.ml_classification.confidence_scores).map(([className, confidence]) => (
                        <Grid item xs={6} md={4} key={className}>
                          <Paper sx={{ p: 2 }}>
                            <Typography variant="body2" color="text.secondary">
                              {className.replace('_', ' ').toUpperCase()}
                            </Typography>
                            <Typography variant="h6" color="primary.main">
                              {(confidence * 100).toFixed(1)}%
                            </Typography>
                          </Paper>
                        </Grid>
                      ))}
                    </Grid>
                  </Box>
                )}
                
                {/* Show clinical validation details if available */}
                {results.ml_classification?.clinical_prediction && (
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Clinical Assessment
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={6} md={4}>
                        <Paper sx={{ p: 2 }}>
                          <Typography variant="body2" color="text.secondary">
                            Clinical Classification
                          </Typography>
                          <Typography variant="h6" color="primary.main">
                            {results.ml_classification.clinical_prediction.class || 'N/A'}
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={6} md={4}>
                        <Paper sx={{ p: 2 }}>
                          <Typography variant="body2" color="text.secondary">
                            Clinical Confidence
                          </Typography>
                          <Typography variant="h6" color="primary.main">
                            {((results.ml_classification.clinical_prediction.confidence || 0) * 100).toFixed(1)}%
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={6} md={4}>
                        <Paper sx={{ p: 2 }}>
                          <Typography variant="body2" color="text.secondary">
                            Method Used
                          </Typography>
                          <Typography variant="h6" color="primary.main">
                            {results.ml_classification.clinical_prediction.method || 'N/A'}
                          </Typography>
                        </Paper>
                      </Grid>
                    </Grid>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {activeTab === 2 && (
        <Grid container spacing={3}>
          {/* Step Timeline */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Step Duration Timeline
                </Typography>
                <Box sx={{ height: 300 }}>
                  <Line 
                    data={stepMetricsData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          display: false
                        }
                      }
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Step Metrics Table */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Detailed Step Analysis
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Step #</TableCell>
                        <TableCell>Duration (s)</TableCell>
                        <TableCell>Length (m)</TableCell>
                        <TableCell>Velocity (m/s)</TableCell>
                        <TableCell>Foot</TableCell>
                        <TableCell>Quality</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {results.step_metrics?.slice(0, 10).map((step, index) => (
                        <TableRow key={index}>
                          <TableCell>{index + 1}</TableCell>
                          <TableCell>{step.duration?.toFixed(3) || 'N/A'}</TableCell>
                          <TableCell>{step.length?.toFixed(3) || 'N/A'}</TableCell>
                          <TableCell>{step.velocity?.toFixed(3) || 'N/A'}</TableCell>
                          <TableCell>
                            <Chip 
                              label={step.foot || 'Unknown'} 
                              size="small"
                              color={step.foot === 'Left' ? 'primary' : 'secondary'}
                            />
                          </TableCell>
                          <TableCell>
                            {step.quality > 0.8 ? (
                              <CheckCircle color="success" fontSize="small" />
                            ) : (
                              <Warning color="warning" fontSize="small" />
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {activeTab === 3 && (
        <Grid container spacing={3}>
          {/* Clinical Recommendations */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Clinical Recommendations
                </Typography>
                <Alert 
                  severity={
                    results.ml_classification?.primary_classification === 'Neutral' 
                      ? 'success' 
                      : results.ml_classification?.primary_classification?.includes('Severe')
                      ? 'error'
                      : 'warning'
                  }
                  sx={{ mb: 2 }}
                >
                  Based on your gait pattern ({results.ml_classification?.primary_classification}), 
                  we recommend the following interventions:
                </Alert>
                
                <Grid container spacing={2}>
                  {/* Footwear Recommendations */}
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <LocalHospital color="primary" sx={{ mr: 1 }} />
                        <Typography variant="subtitle1" fontWeight="bold">
                          Footwear
                        </Typography>
                      </Box>
                      <Typography variant="body2" paragraph>
                        {(() => {
                          const classification = results.ml_classification?.primary_classification || '';
                          if (classification.includes('Pronation') || classification.includes('Overpronation')) {
                            if (classification.includes('Severe')) {
                              return 'Motion control shoes with maximum stability features and firm midsoles. Consider custom orthotics.';
                            } else if (classification.includes('Moderate')) {
                              return 'Stability shoes with moderate motion control and structured support.';
                            } else if (classification.includes('Mild')) {
                              return 'Light stability shoes with mild motion control features.';
                            } else {
                              return 'Motion control or stability shoes with firm midsoles';
                            }
                          } else if (classification === 'Supination') {
                            return 'Neutral cushioned shoes with flexibility and extra cushioning';
                          } else {
                            return 'Neutral running shoes with moderate cushioning';
                          }
                        })()}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Consider custom orthotics for additional support
                      </Typography>
                    </Paper>
                  </Grid>

                  {/* Exercise Recommendations */}
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <Speed color="primary" sx={{ mr: 1 }} />
                        <Typography variant="subtitle1" fontWeight="bold">
                          Exercises
                        </Typography>
                      </Box>
                      <Typography variant="body2" paragraph>
                        • Calf raises for ankle strength
                        <br />• Toe walks for arch support
                        <br />• Single-leg balance exercises
                        <br />• Resistance band exercises
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Perform 3 sets of 15 reps daily
                      </Typography>
                    </Paper>
                  </Grid>

                  {/* Follow-up Recommendations */}
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <Psychology color="primary" sx={{ mr: 1 }} />
                        <Typography variant="subtitle1" fontWeight="bold">
                          Follow-up
                        </Typography>
                      </Box>
                      <Typography variant="body2" paragraph>
                        {results.ml_classification?.primary_classification?.includes('Severe')
                          ? 'Consult a podiatrist or orthopedic specialist'
                          : results.ml_classification?.primary_classification?.includes('Moderate')
                          ? 'Schedule follow-up analysis in 4-6 weeks'
                          : 'Re-assess gait patterns in 3 months'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Monitor for pain or discomfort
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>

                {/* Dynamic Risk Assessment */}
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Risk Assessment
                  </Typography>
                  <Grid container spacing={1}>
                    {(() => {
                      const classification = results.ml_classification?.primary_classification || '';
                      const confidence = results.ml_classification?.confidence || 0;
                      const asymmetryIndex = results.biomechanical_profile?.asymmetry_analysis?.degree_difference || 0;
                      const efficiencyScore = results.biomechanical_metrics?.efficiency_score || 80;
                      const symmetryScore = results.biomechanical_metrics?.symmetry_score || 85;

                      // Calculate dynamic risk levels
                      const getInjuryRisk = () => {
                        if (classification.includes('Severe')) return { label: 'Injury Risk: High', color: 'error' };
                        if (classification.includes('Moderate')) return { label: 'Injury Risk: Moderate', color: 'warning' };
                        if (classification.includes('Mild')) return { label: 'Injury Risk: Low-Moderate', color: 'info' };
                        return { label: 'Injury Risk: Low', color: 'success' };
                      };

                      const getBalanceRisk = () => {
                        if (asymmetryIndex > 15) return { label: 'Balance: Poor', color: 'error' };
                        if (asymmetryIndex > 10) return { label: 'Balance: Fair', color: 'warning' };
                        if (asymmetryIndex > 5) return { label: 'Balance: Good', color: 'info' };
                        return { label: 'Balance: Excellent', color: 'success' };
                      };

                      const getEfficiency = () => {
                        if (efficiencyScore < 60) return { label: 'Efficiency: Poor', color: 'error' };
                        if (efficiencyScore < 75) return { label: 'Efficiency: Fair', color: 'warning' };
                        if (efficiencyScore < 85) return { label: 'Efficiency: Good', color: 'info' };
                        return { label: 'Efficiency: Excellent', color: 'success' };
                      };

                      const getSymmetry = () => {
                        if (symmetryScore < 60) return { label: 'Symmetry: Poor', color: 'error' };
                        if (symmetryScore < 75) return { label: 'Symmetry: Fair', color: 'warning' };
                        if (symmetryScore < 85) return { label: 'Symmetry: Good', color: 'info' };
                        return { label: 'Symmetry: Excellent', color: 'success' };
                      };

                      const risks = [getInjuryRisk(), getBalanceRisk(), getEfficiency(), getSymmetry()];

                      return risks.map((risk, index) => (
                        <Grid item xs={12} sm={6} md={3} key={index}>
                          <Chip 
                            label={risk.label} 
                            color={risk.color} 
                            variant="outlined"
                            sx={{ width: '100%' }}
                          />
                        </Grid>
                      ));
                    })()}
                  </Grid>
                  
                  {/* Risk Details */}
                  {(() => {
                    const classification = results.ml_classification?.primary_classification || '';
                    if (classification.includes('Severe')) {
                      return (
                        <Alert severity="error" sx={{ mt: 2 }}>
                          <Typography variant="body2">
                            <strong>High Risk Factors:</strong> Severe gait abnormalities increase risk of injury, joint stress, and long-term complications. 
                            Immediate consultation with a healthcare provider is recommended.
                          </Typography>
                        </Alert>
                      );
                    } else if (classification.includes('Moderate')) {
                      return (
                        <Alert severity="warning" sx={{ mt: 2 }}>
                          <Typography variant="body2">
                            <strong>Moderate Risk Factors:</strong> Consider corrective footwear and exercises to prevent progression. 
                            Monitor for pain or discomfort during activities.
                          </Typography>
                        </Alert>
                      );
                    } else if (classification.includes('Mild')) {
                      return (
                        <Alert severity="info" sx={{ mt: 2 }}>
                          <Typography variant="body2">
                            <strong>Low Risk:</strong> Minor gait variations detected. Preventive measures and regular monitoring recommended.
                          </Typography>
                        </Alert>
                      );
                    } else {
                      return (
                        <Alert severity="success" sx={{ mt: 2 }}>
                          <Typography variant="body2">
                            <strong>Normal Gait Pattern:</strong> Continue current activities and maintain regular exercise routine.
                          </Typography>
                        </Alert>
                      );
                    }
                  })()}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Additional Notes */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Important Notes
                </Typography>
                <Alert severity="info" sx={{ mb: 1 }}>
                  This analysis is for informational purposes only and should not replace professional medical advice.
                </Alert>
                <Typography variant="body2" paragraph>
                  • Results are based on video analysis and machine learning algorithms
                  • Accuracy depends on video quality and recording conditions
                  • Consult a healthcare professional for persistent issues
                  • Regular monitoring can help track improvements
                </Typography>
                <Divider sx={{ my: 2 }} />
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Button 
                    variant="contained" 
                    onClick={() => navigate('/multi-angle')}
                    startIcon={<Analytics />}
                  >
                    New Analysis
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {activeTab === 4 && showVisualizationTab && (
        <Grid container spacing={3}>
          {/* Visualization Videos */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Gait Analysis Videos with Skeletal Tracking
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  These videos show the detected leg movements and joint tracking used for analysis.
                </Typography>
                
                {visualizations.length === 0 ? (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <CircularProgress />
                    <Typography variant="body2" sx={{ mt: 2 }}>
                      Processing visualization videos...
                    </Typography>
                  </Box>
                ) : (
                  <Grid container spacing={3}>
                    {visualizations.map((viz, index) => (
                      <Grid item xs={12} md={6} lg={4} key={index}>
                        <Card variant="outlined">
                          <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                              <Videocam color="primary" sx={{ mr: 1 }} />
                              <Typography variant="h6">
                                {viz.angle.charAt(0).toUpperCase() + viz.angle.slice(1)} View
                              </Typography>
                            </Box>
                            
                            {/* Video Player */}
                            <Box sx={{ mb: 2, position: 'relative' }}>
                              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                Video URL: {config.getApiUrl(viz.visualization_url)}
                              </Typography>
                              <video 
                                key={viz.visualization_url} // Force re-render
                                width="100%" 
                                height="auto" 
                                controls
                                preload="metadata"
                                muted // Add muted for autoplay policies
                                playsInline // Better mobile support
                                style={{ 
                                  borderRadius: '8px', 
                                  backgroundColor: '#000', 
                                  minHeight: '200px',
                                  border: '1px solid #ddd'
                                }}
                                onError={(e) => {
                                  console.error('Video error:', e);
                                  console.error('Video error details:', {
                                    error: e.target.error,
                                    networkState: e.target.networkState,
                                    readyState: e.target.readyState,
                                    src: e.target.currentSrc
                                  });
                                  e.target.style.display = 'none';
                                  e.target.nextSibling.style.display = 'block';
                                }}
                                onLoadStart={() => console.log('Video loading started for:', viz.visualization_url)}
                                onCanPlay={() => console.log('Video can play')}
                                onLoadedMetadata={() => console.log('Video metadata loaded')}
                                onClick={(e) => {
                                  console.log('Video clicked');
                                  if (e.target.paused) {
                                    e.target.play();
                                  } else {
                                    e.target.pause();
                                  }
                                }}
                              >
                                <source 
                                  src={config.getApiUrl(viz.visualization_url)} 
                                  type="video/mp4" 
                                />
                                Your browser does not support the video tag.
                              </video>
                              
                              {/* Fallback message */}
                              <Box
                                sx={{
                                  display: 'none',
                                  textAlign: 'center',
                                  p: 4,
                                  bgcolor: 'grey.100',
                                  borderRadius: '8px',
                                  border: '2px dashed',
                                  borderColor: 'grey.300'
                                }}
                              >
                                <Videocam sx={{ fontSize: 48, color: 'grey.400', mb: 2 }} />
                                <Typography variant="body2" color="text.secondary">
                                  Video format not supported by your browser
                                </Typography>
                                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                  Try downloading the video to view it in a media player
                                </Typography>
                                <Button
                                  variant="outlined"
                                  size="small"
                                  onClick={() => {
                                    const link = document.createElement('a');
                                    link.href = config.getApiUrl(viz.visualization_url);
                                    link.download = viz.filename;
                                    link.click();
                                  }}
                                >
                                  Download Video
                                </Button>
                              </Box>
                            </Box>
                            
                            {/* Video Info */}
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                              <Typography variant="body2" color="text.secondary">
                                Detection Rate: {((viz.metrics?.detection_rate || 0) * 100).toFixed(1)}%
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                {viz.metrics?.frame_count || 0} frames
                              </Typography>
                            </Box>
                            
                            {/* Metrics Summary */}
                            {viz.metrics && (
                              <Box>
                                <Typography variant="subtitle2" gutterBottom>
                                  Analysis Summary:
                                </Typography>
                                {viz.metrics.avg_pronation && (
                                  <Typography variant="body2">
                                    • Avg Pronation: {viz.metrics.avg_pronation.toFixed(1)}°
                                  </Typography>
                                )}
                                {viz.metrics.avg_knee_flexion && (
                                  <Typography variant="body2">
                                    • Avg Knee Flexion: {viz.metrics.avg_knee_flexion.toFixed(1)}°
                                  </Typography>
                                )}
                                <Typography variant="body2">
                                  • Frames Analyzed: {viz.metrics.frames_with_detection}
                                </Typography>
                              </Box>
                            )}
                            
                            {/* Legend */}
                            <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                              <Typography variant="caption" display="block" gutterBottom>
                                <strong>Color Legend:</strong>
                              </Typography>
                              <Typography variant="caption" display="block" sx={{ color: 'blue' }}>
                                🔵 Blue Lines: Left Leg
                              </Typography>
                              <Typography variant="caption" display="block" sx={{ color: 'green' }}>
                                🟢 Green Lines: Right Leg
                              </Typography>
                              <Typography variant="caption" display="block" sx={{ color: 'orange' }}>
                                🟡 White Circles: Joint Points
                              </Typography>
                            </Box>
                            
                            {/* Action Buttons */}
                            <Box sx={{ mt: 2, textAlign: 'center', display: 'flex', gap: 1, justifyContent: 'center' }}>
                              <Button
                                variant="outlined"
                                startIcon={<PlayArrow />}
                                size="small"
                                onClick={() => {
                                  window.open(config.getApiUrl(viz.visualization_url), '_blank');
                                }}
                              >
                                Open in New Tab
                              </Button>
                              <Button
                                variant="outlined"
                                startIcon={<Download />}
                                size="small"
                                onClick={() => {
                                  const link = document.createElement('a');
                                  link.href = config.getApiUrl(viz.visualization_url);
                                  link.download = viz.filename;
                                  link.click();
                                }}
                              >
                                Download
                              </Button>
                            </Box>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                )}
              </CardContent>
            </Card>
          </Grid>
          
          {/* Instructions */}
          <Grid item xs={12}>
            <Alert severity="info">
              <Typography variant="body2">
                <strong>How to interpret the videos:</strong>
              </Typography>
              <Typography variant="body2" component="div" sx={{ mt: 1 }}>
                • The colored lines show the AI's detection of your leg segments and joints<br/>
                • Real-time metrics are displayed in the top-left corner<br/>
                • The progress bar shows the analysis completion<br/>
                • Green pronation values (-5° to +5°) indicate neutral gait<br/>
                • Higher positive values indicate pronation (foot rolling inward)
              </Typography>
            </Alert>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default AdvancedAnalysis;