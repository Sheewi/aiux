#!/bin/bash
# Microagents Conversational AI Demo Launcher
# Run from external drive

echo "🤖 Starting Microagents Conversational AI Demo..."
echo "📍 Running from: $(pwd)"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Run the demo
python3 demo_conversational_ai.py
