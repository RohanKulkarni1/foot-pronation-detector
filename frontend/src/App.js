import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box, AppBar, Toolbar, Typography, Container } from '@mui/material';

// Import Phase 2 components
import HomePage from './pages/HomePage';
import MultiAngleCapture from './pages/MultiAngleCapture';
import AdvancedAnalysis from './pages/AdvancedAnalysis';
import MLDashboard from './pages/MLDashboard';
import DatasetManager from './pages/DatasetManager';
import Navigation from './components/Navigation';

// Create enhanced theme for Phase 2
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 500,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          borderRadius: '12px',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: '8px',
          fontWeight: 500,
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ flexGrow: 1 }}>
          <AppBar position="static" elevation={1}>
            <Toolbar>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                Foot Analysis Phase 2 - Advanced Gait Analysis
              </Typography>
              <Typography variant="body2" sx={{ mr: 2 }}>
                v2.0.0
              </Typography>
            </Toolbar>
          </AppBar>
          
          <Navigation />
          
          <Container maxWidth="xl" sx={{ mt: 3, mb: 3 }}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/capture" element={<MultiAngleCapture />} />
              <Route path="/analysis/:sessionId?" element={<AdvancedAnalysis />} />
              <Route path="/ml-dashboard" element={<MLDashboard />} />
              <Route path="/datasets" element={<DatasetManager />} />
            </Routes>
          </Container>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;