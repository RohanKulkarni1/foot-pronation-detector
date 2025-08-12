import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  Tabs,
  Tab,
  Paper
} from '@mui/material';
import {
  Home,
  VideoCall,
  Analytics,
  Psychology,
  Storage
} from '@mui/icons-material';

const Navigation = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const tabs = [
    { label: 'Home', icon: <Home />, path: '/' },
    { label: 'Multi-Angle Capture', icon: <VideoCall />, path: '/capture' },
    { label: 'Advanced Analysis', icon: <Analytics />, path: '/analysis' },
    { label: 'ML Dashboard', icon: <Psychology />, path: '/ml-dashboard' },
    { label: 'Dataset Manager', icon: <Storage />, path: '/datasets' }
  ];

  const getCurrentTab = () => {
    const currentPath = location.pathname;
    const tabIndex = tabs.findIndex(tab => 
      tab.path === currentPath || (tab.path !== '/' && currentPath.startsWith(tab.path))
    );
    return tabIndex >= 0 ? tabIndex : 0;
  };

  const handleTabChange = (event, newValue) => {
    navigate(tabs[newValue].path);
  };

  return (
    <Paper elevation={1} sx={{ borderRadius: 0 }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs 
          value={getCurrentTab()} 
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
          sx={{ minHeight: '48px' }}
        >
          {tabs.map((tab, index) => (
            <Tab
              key={index}
              icon={tab.icon}
              label={tab.label}
              iconPosition="start"
              sx={{ 
                minHeight: '48px',
                textTransform: 'none',
                fontSize: '0.875rem'
              }}
            />
          ))}
        </Tabs>
      </Box>
    </Paper>
  );
};

export default Navigation;