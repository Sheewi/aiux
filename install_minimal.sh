#!/bin/bash
# Quick minimal installation for testing
# This installs only the core packages needed

echo "🚀 Minimal Installation for Core Testing"
echo "========================================"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️ No virtual environment detected. Creating one..."
    python3 -m venv ~/venv/grok-minimal
    source ~/venv/grok-minimal/bin/activate
    echo "✅ Activated: ~/venv/grok-minimal"
else
    echo "✅ Using existing virtual environment: $VIRTUAL_ENV"
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install minimal requirements
echo "📥 Installing minimal requirements..."
pip install -r requirements-minimal.txt --no-cache-dir

# Test the installation
echo "🧪 Testing minimal installation..."
python -c "
import sys
packages = ['click', 'redis', 'requests', 'fastapi', 'pydantic', 'pandas', 'numpy']
failed = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')
        failed.append(pkg)

if failed:
    print(f'\\n❌ Failed: {failed}')
    sys.exit(1)
else:
    print('\\n🎉 All minimal packages installed successfully!')
"

if [ $? -eq 0 ]; then
    echo "✅ Minimal installation complete!"
    echo "To test: python test_imports.py"
else
    echo "❌ Minimal installation failed"
    echo "Try individual package installation:"
    echo "pip install click==8.1.7"
    echo "pip install redis==5.0.1"
    echo "pip install requests pandas numpy"
fi
