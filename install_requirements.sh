#!/bin/bash
# Installation script for Grok UI requirements
# Recommended installation method with known-working versions

echo "🚀 Setting up Grok UI Environment"
echo "=================================="

# Create fresh virtual environment
echo "📦 Creating fresh virtual environment..."
python3 -m venv ~/venv/grok-ui

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source ~/venv/grok-ui/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Clear any existing cache
echo "🧹 Clearing pip cache..."
pip cache purge

# Install requirements with legacy resolver for compatibility
echo "📥 Installing requirements with legacy resolver..."
pip install -r requirements.txt --use-deprecated=legacy-resolver

# Verify installation
echo "✅ Verifying key packages..."
python -c "
import sys
packages = ['click', 'redis', 'APScheduler', 'paramiko', 'fabric']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg} installed successfully')
    except ImportError as e:
        print(f'✗ {pkg} failed: {e}')
"

echo "🎉 Installation complete!"
echo "To activate this environment in the future, run:"
echo "source ~/venv/grok-ui/bin/activate"
