import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider
} from '@mui/material';
import {
  VideoCall,
  Analytics,
  Psychology,
  Storage,
  CheckCircle,
  ArrowForward,
  Upgrade
} from '@mui/icons-material';

const HomePage = () => {
  const navigate = useNavigate();

  const features = [
    {
      title: 'Multi-Angle Video Capture',
      description: 'Record from multiple camera angles simultaneously for comprehensive 3D gait analysis',
      icon: <VideoCall sx={{ fontSize: 40 }} />,
      action: () => navigate('/capture'),
      buttonText: 'Start Recording',
      color: 'primary'
    },
    {
      title: 'Advanced ML Analysis',
      description: 'AI-powered classification using ensemble models and neural networks',
      icon: <Psychology sx={{ fontSize: 40 }} />,
      action: () => navigate('/ml-dashboard'),
      buttonText: 'View ML Dashboard',
      color: 'secondary'
    },
    {
      title: 'Enhanced Analytics',
      description: 'Detailed biomechanical analysis with step detection and asymmetry assessment',
      icon: <Analytics sx={{ fontSize: 40 }} />,
      action: () => navigate('/analysis'),
      buttonText: 'View Analytics',
      color: 'success'
    },
    {
      title: 'Dataset Management',
      description: 'Manage training datasets and improve model accuracy with labeled data',
      icon: <Storage sx={{ fontSize: 40 }} />,
      action: () => navigate('/datasets'),
      buttonText: 'Manage Data',
      color: 'warning'
    }
  ];

  const phase2Enhancements = [
    'Multi-angle synchronized video recording',
    'Enhanced biomechanical feature extraction',
    'ML ensemble classification (RF, GB, Neural Networks)',
    'Step detection and gait cycle analysis',
    'Asymmetry and efficiency calculations',
    '3D pose reconstruction capabilities',
    'Clinical validation framework',
    'Professional reporting system'
  ];

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography variant="h3" gutterBottom sx={{ fontWeight: 700, color: 'primary.main' }}>
          Advanced Gait Analysis Platform
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
          Phase 2: Enhanced ML-Powered Biomechanical Assessment
        </Typography>
        <Chip 
          label="Phase 2.0.0" 
          color="primary" 
          icon={<Upgrade />}
          sx={{ fontSize: '0.9rem', fontWeight: 500 }}
        />
      </Box>

      {/* Features Grid */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {features.map((feature, index) => (
          <Grid item xs={12} md={6} key={index}>
            <Card 
              sx={{ 
                height: '100%', 
                display: 'flex', 
                flexDirection: 'column',
                transition: 'transform 0.2s, boxShadow 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: 4
                }
              }}
            >
              <CardContent sx={{ flexGrow: 1, textAlign: 'center', pt: 3 }}>
                <Box sx={{ color: `${feature.color}.main`, mb: 2 }}>
                  {feature.icon}
                </Box>
                <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
                  {feature.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {feature.description}
                </Typography>
              </CardContent>
              <CardActions sx={{ justifyContent: 'center', pb: 3 }}>
                <Button
                  variant="contained"
                  color={feature.color}
                  onClick={feature.action}
                  endIcon={<ArrowForward />}
                  sx={{ borderRadius: 2 }}
                >
                  {feature.buttonText}
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Phase 2 Enhancements */}
      <Card sx={{ mt: 4 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, color: 'primary.main' }}>
            Phase 2 Advanced Features
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
            Enhanced capabilities beyond Phase 1 basic gait analysis
          </Typography>
          <Divider sx={{ mb: 2 }} />
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <List dense>
                {phase2Enhancements.slice(0, 4).map((enhancement, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <CheckCircle color="success" />
                    </ListItemIcon>
                    <ListItemText primary={enhancement} />
                  </ListItem>
                ))}
              </List>
            </Grid>
            <Grid item xs={12} md={6}>
              <List dense>
                {phase2Enhancements.slice(4).map((enhancement, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <CheckCircle color="success" />
                    </ListItemIcon>
                    <ListItemText primary={enhancement} />
                  </ListItem>
                ))}
              </List>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Quick Start */}
      <Card sx={{ mt: 3, background: 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)' }}>
        <CardContent sx={{ textAlign: 'center', color: 'white' }}>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
            Ready to Get Started?
          </Typography>
          <Typography variant="body1" sx={{ mb: 3 }}>
            Begin with multi-angle video capture for the most comprehensive gait analysis
          </Typography>
          <Button
            variant="contained"
            color="inherit"
            size="large"
            onClick={() => navigate('/capture')}
            endIcon={<VideoCall />}
            sx={{ 
              backgroundColor: 'white', 
              color: 'primary.main',
              '&:hover': {
                backgroundColor: 'rgba(255,255,255,0.9)'
              }
            }}
          >
            Start Multi-Angle Capture
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
};

export default HomePage;