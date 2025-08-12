#!/bin/bash

# Phase 2 Startup Script
echo "Starting Foot Analysis Phase 2..."

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Function to check if port is in use
check_port() {
    lsof -i :$1 > /dev/null 2>&1
    return $?
}

# Function to kill processes on port
kill_port() {
    echo "Killing processes on port $1..."
    pkill -f "port.*$1" 2>/dev/null || true
    lsof -ti:$1 | xargs kill -9 2>/dev/null || true
    sleep 2
}

# Check and kill existing processes on Phase 2 ports
if check_port 8001; then
    echo "Port 8001 is in use, killing existing processes..."
    kill_port 8001
fi

if check_port 3001; then
    echo "Port 3001 is in use, killing existing processes..."
    kill_port 3001
fi

# Start backend server
echo "Starting Phase 2 backend server on port 8001..."
cd "$PROJECT_DIR/backend"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt > /dev/null 2>&1

# Start backend in background
echo "Starting FastAPI backend..."
python3 -m uvicorn app:app --host 0.0.0.0 --port 8001 --reload > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID (port 8001)"

# Wait for backend to start
sleep 5

# Check if backend is running
if curl -s http://localhost:8001/health > /dev/null; then
    echo "✅ Backend health check passed"
else
    echo "❌ Backend health check failed"
fi

# Start frontend server
echo "Starting Phase 2 frontend server on port 3001..."
cd "$PROJECT_DIR/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi

# Start frontend in background
echo "Starting React frontend..."
mkdir -p ../logs
BROWSER=none PORT=3001 npm start > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID (port 3001)"

# Wait for frontend to compile
echo "Waiting for frontend to compile..."
sleep 15

# Check if frontend is running
if curl -s -I http://localhost:3001 > /dev/null; then
    echo "✅ Frontend server is running"
else
    echo "❌ Frontend server check failed"
fi

# Create PID file for easy cleanup
echo "$BACKEND_PID" > "$PROJECT_DIR/logs/backend.pid"
echo "$FRONTEND_PID" > "$PROJECT_DIR/logs/frontend.pid"

# Display status
echo ""
echo "🚀 Phase 2 Application Started Successfully!"
echo "📊 Backend API: http://localhost:8001"
echo "🖥️  Frontend:    http://localhost:3001"
echo "📋 Health:      http://localhost:8001/health"
echo ""
echo "💡 To stop the application, run: ./scripts/stop-all.sh"
echo "📁 Logs are available in: logs/"
echo ""

# Open browser
if command -v open &> /dev/null; then
    echo "Opening application in browser..."
    sleep 2
    open http://localhost:3001
fi

echo "Phase 2 startup complete! 🎉"