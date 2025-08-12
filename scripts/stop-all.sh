#!/bin/bash

# Stop All Servers Script
echo "Stopping all Foot Analysis servers..."

# Function to kill processes on port
kill_port() {
    local port=$1
    local name=$2
    
    echo "Stopping $name on port $port..."
    
    # Kill by port
    lsof -ti:$port | xargs kill -9 2>/dev/null || true
    
    # Kill by process name patterns
    case $port in
        3000|3001)
            pkill -f "react-scripts start" 2>/dev/null || true
            pkill -f "node.*$port" 2>/dev/null || true
            ;;
        8000|8001)
            pkill -f "uvicorn.*$port" 2>/dev/null || true
            pkill -f "python.*app:app.*$port" 2>/dev/null || true
            ;;
    esac
    
    sleep 1
    
    # Check if still running
    if lsof -i :$port > /dev/null 2>&1; then
        echo "⚠️  Warning: $name on port $port may still be running"
    else
        echo "✅ $name stopped"
    fi
}

# Stop Phase 1 servers (ports 3000, 8000)
kill_port 3000 "Phase 1 Frontend"
kill_port 8000 "Phase 1 Backend"

# Stop Phase 2 servers (ports 3001, 8001)
kill_port 3001 "Phase 2 Frontend"
kill_port 8001 "Phase 2 Backend"

# Additional cleanup
echo "Performing additional cleanup..."

# Kill any remaining Node.js and Python processes related to our apps
pkill -f "foot-analysis" 2>/dev/null || true
pkill -f "Foot.*Analysis" 2>/dev/null || true

# Clean up PID files if they exist
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_DIR/logs/backend.pid" ]; then
    PID=$(cat "$PROJECT_DIR/logs/backend.pid" 2>/dev/null)
    if [ ! -z "$PID" ]; then
        kill -9 "$PID" 2>/dev/null || true
    fi
    rm -f "$PROJECT_DIR/logs/backend.pid"
fi

if [ -f "$PROJECT_DIR/logs/frontend.pid" ]; then
    PID=$(cat "$PROJECT_DIR/logs/frontend.pid" 2>/dev/null)
    if [ ! -z "$PID" ]; then
        kill -9 "$PID" 2>/dev/null || true
    fi
    rm -f "$PROJECT_DIR/logs/frontend.pid"
fi

echo ""
echo "🛑 All servers stopped!"
echo ""
echo "Available commands:"
echo "  ./scripts/start-phase1.sh  - Start Phase 1 (Basic Analysis)"
echo "  ./scripts/start-phase2.sh  - Start Phase 2 (Advanced ML Analysis)"
echo ""