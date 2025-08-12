#!/bin/bash

# Phase 1 Startup Script
echo "Starting Foot Analysis Phase 1..."

# Get the script directory and find Phase 1 directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PHASE1_DIR="$PROJECT_DIR/../Foot Analysis"

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

# Check if Phase 1 directory exists
if [ ! -d "$PHASE1_DIR" ]; then
    echo "❌ Phase 1 directory not found: $PHASE1_DIR"
    exit 1
fi

# Check and kill existing processes on Phase 1 ports
if check_port 8000; then
    echo "Port 8000 is in use, killing existing processes..."
    kill_port 8000
fi

if check_port 3000; then
    echo "Port 3000 is in use, killing existing processes..."
    kill_port 3000
fi

# Start Phase 1 backend
echo "Starting Phase 1 backend server on port 8000..."
cd "$PHASE1_DIR/backend"

echo "Starting FastAPI backend..."
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID (port 8000)"

# Wait for backend to start
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Phase 1 Backend health check passed"
else
    echo "❌ Phase 1 Backend health check failed"
fi

# Start Phase 1 frontend
echo "Starting Phase 1 frontend server on port 3000..."
cd "$PHASE1_DIR/frontend"

echo "Starting React frontend..."
BROWSER=none npm start > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID (port 3000)"

# Wait for frontend to compile
echo "Waiting for frontend to compile..."
sleep 15

# Check if frontend is running
if curl -s -I http://localhost:3000 > /dev/null; then
    echo "✅ Phase 1 Frontend server is running"
else
    echo "❌ Phase 1 Frontend server check failed"
fi

# Display status
echo ""
echo "🚀 Phase 1 Application Started Successfully!"
echo "📊 Backend API: http://localhost:8000"
echo "🖥️  Frontend:    http://localhost:3000"
echo "📋 Health:      http://localhost:8000/health"
echo ""
echo "💡 To stop the application, run: ./scripts/stop-all.sh"
echo ""

# Open browser
if command -v open &> /dev/null; then
    echo "Opening Phase 1 application in browser..."
    sleep 2
    open http://localhost:3000
fi

echo "Phase 1 startup complete! 🎉"