#!/bin/bash
# Universal AI System - Complete Installation Script
# Installs all dependencies for the integrated AI system

set -e  # Exit on any error

echo "🚀 Universal AI System - Complete Installation"
echo "=============================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
echo "📋 Python version: $python_version"

if [ "$(printf '%s\n' "3.8" "$python_version" | sort -V | head -n1)" != "3.8" ]; then
    echo "❌ Error: Python 3.8 or higher required"
    exit 1
fi

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  Warning: No virtual environment detected. Consider using:"
    echo "   python3 -m venv venv && source venv/bin/activate"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install main requirements
echo "📥 Installing Universal AI System requirements..."
pip install -r requirements.txt

# Install microagents requirements  
echo "📥 Installing microagents conversational AI requirements..."
pip install -r microagents_conversational_ai/installs/requirements.txt

# Install tokenizer requirements
echo "📥 Installing tokenizer requirements..."
pip install -r tokenizer/requirements.txt

# Platform-specific installations
echo "🔧 Installing platform-specific dependencies..."

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)
        echo "🐧 Linux detected - installing Linux-specific packages..."
        
        # Install system packages (requires sudo)
        if command -v apt-get >/dev/null 2>&1; then
            echo "📦 Installing apt packages..."
            sudo apt-get update
            sudo apt-get install -y \
                libudev-dev \
                libusb-1.0-0-dev \
                portaudio19-dev \
                python3-dev \
                build-essential \
                pkg-config \
                libasound2-dev \
                libpulse-dev
            
            # Install optional packages
            echo "📦 Installing optional Linux packages..."
            pip install evdev>=1.6.0 || echo "⚠️ evdev installation failed (normal on non-Linux systems)"
            
        elif command -v yum >/dev/null 2>&1; then
            echo "📦 Installing yum packages..."
            sudo yum update
            sudo yum install -y \
                libudev-devel \
                libusb1-devel \
                portaudio-devel \
                python3-devel \
                gcc \
                gcc-c++ \
                pkgconfig \
                alsa-lib-devel \
                pulseaudio-libs-devel
        fi
        ;;
    Darwin*)
        echo "🍎 macOS detected - installing macOS-specific packages..."
        
        # Install with Homebrew if available
        if command -v brew >/dev/null 2>&1; then
            echo "🍺 Installing Homebrew packages..."
            brew install portaudio libusb pkg-config
        else
            echo "⚠️ Homebrew not found. Please install manually:"
            echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        fi
        ;;
    CYGWIN*|MINGW32*|MSYS*|MINGW*)
        echo "🪟 Windows detected - installing Windows-specific packages..."
        echo "⚠️ Note: Some hardware packages may require additional setup on Windows"
        ;;
    *)
        echo "❓ Unknown OS: ${OS}"
        echo "⚠️ Some hardware packages may not be available"
        ;;
esac

# Install Google Cloud SDK (optional)
echo "☁️ Checking Google Cloud SDK..."
if command -v gcloud >/dev/null 2>&1; then
    echo "✅ Google Cloud SDK already installed"
    gcloud version
else
    echo "⚠️ Google Cloud SDK not found. For production deployment, install from:"
    echo "   https://cloud.google.com/sdk/docs/install"
fi

# Verify critical imports
echo "🧪 Verifying critical package imports..."

python3 -c "
import sys
import importlib

packages = [
    ('asyncio', 'Core async support'),
    ('aiohttp', 'Async HTTP'),
    ('fastapi', 'Web framework'),
    ('psutil', 'System utilities'),
    ('numpy', 'Numerical computing'),
    ('pandas', 'Data processing')
]

hardware_packages = [
    ('pyudev', 'USB device discovery'),
    ('serial', 'Serial communication'),
    ('sounddevice', 'Audio devices'),
    ('cv2', 'Computer vision')
]

optional_packages = [
    ('playwright', 'Browser automation'),
    ('scrapy', 'Web scraping'),
    ('kubernetes', 'Container orchestration'),
    ('docker', 'Container management')
]

def check_package(name, description):
    try:
        importlib.import_module(name)
        print(f'✅ {name:15} - {description}')
        return True
    except ImportError:
        print(f'❌ {name:15} - {description} (MISSING)')
        return False

print('Core packages:')
core_ok = all(check_package(pkg, desc) for pkg, desc in packages)

print('\nHardware packages:')
hardware_ok = sum(check_package(pkg, desc) for pkg, desc in hardware_packages)

print('\nOptional packages:')
optional_ok = sum(check_package(pkg, desc) for pkg, desc in optional_packages)

print(f'\n📊 Installation Summary:')
print(f'  Core packages: {\"✅ All OK\" if core_ok else \"❌ Some missing\"}')
print(f'  Hardware packages: {hardware_ok}/{len(hardware_packages)} available')
print(f'  Optional packages: {optional_ok}/{len(optional_packages)} available')

if not core_ok:
    print('❌ Critical packages missing - check installation')
    sys.exit(1)
else:
    print('✅ Core installation successful!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "🚀 Next steps:"
    echo "  1. Test the system: python3 main_orchestrator.py"
    echo "  2. Test integration: python3 integrated_ai_system.py"
    echo "  3. For production: Configure Google Cloud credentials"
    echo ""
    echo "📚 Documentation:"
    echo "  • README.md - Complete system guide"
    echo "  • IMPLEMENTATION_COMPLETE.md - Project summary"
    echo "  • MISSING_COMPONENTS_ANALYSIS.md - Component status"
    echo ""
else
    echo "❌ Installation completed with errors. Check the output above."
    exit 1
fi
