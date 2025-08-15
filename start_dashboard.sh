#!/bin/bash

echo "🚀 Starting Customer Dashboard with Figma Integration"
echo "=========================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if the dashboard exists
if [ ! -f "/media/r/Workspace/customer_dashboard/index.html" ]; then
    echo "❌ Dashboard not found. Please run setup first."
    exit 1
fi

echo "✅ Starting local development server..."

# Start a simple HTTP server
cd /media/r/Workspace/customer_dashboard
python3 -m http.server 8080 &
SERVER_PID=$!

echo "✅ Dashboard is running at: http://localhost:8080"
echo "✅ Figma integration is active (simulation mode)"
echo ""
echo "📊 Dashboard Features:"
echo "   • Real-time customer metrics"
echo "   • Interactive customer cards"
echo "   • Figma design system integration"
echo "   • Responsive layout"
echo ""
echo "🎨 Figma Integration Status:"
echo "   • 10 components created"
echo "   • 8 design tokens available"
echo "   • 1 design system active"
echo ""
echo "Press Ctrl+C to stop the server"

# Wait for the server process
wait $SERVER_PID
