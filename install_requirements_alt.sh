#!/bin/bash
# Alternative installation script with system dependencies
# For systems with complex dependency requirements

echo "🔧 Installing System Dependencies and Requirements"
echo "================================================"

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt &> /dev/null; then
            echo "ubuntu"
        elif command -v yum &> /dev/null; then
            echo "rhel"
        elif command -v pacman &> /dev/null; then
            echo "arch"
        else
            echo "unknown-linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Install system dependencies based on OS
install_system_deps() {
    local os=$(detect_os)
    echo "🔍 Detected OS: $os"
    
    case $os in
        "ubuntu")
            echo "📦 Installing Ubuntu/Debian system dependencies..."
            sudo apt update
            sudo apt install -y python3-dev libffi-dev libssl-dev build-essential
            ;;
        "rhel")
            echo "📦 Installing RHEL/CentOS system dependencies..."
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y python3-devel libffi-devel openssl-devel
            ;;
        "arch")
            echo "📦 Installing Arch Linux system dependencies..."
            sudo pacman -S --noconfirm base-devel python libffi openssl
            ;;
        "macos")
            echo "📦 Installing macOS dependencies with Homebrew..."
            if ! command -v brew &> /dev/null; then
                echo "❌ Homebrew not found. Please install Homebrew first."
                exit 1
            fi
            brew install libffi openssl
            ;;
        *)
            echo "⚠️ Unknown OS. Please install development tools manually."
            ;;
    esac
}

# Main installation function
main() {
    echo "🚀 Starting alternative installation process..."
    
    # Install system dependencies
    install_system_deps
    
    # Create fresh virtual environment
    echo "📦 Creating fresh virtual environment..."
    python3 -m venv ~/venv/grok-ui-alt
    
    # Activate virtual environment
    echo "🔄 Activating virtual environment..."
    source ~/venv/grok-ui-alt/bin/activate
    
    # Upgrade pip
    echo "⬆️ Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    # Clear cache
    echo "🧹 Clearing pip cache..."
    pip cache purge
    
    # Install with ignore-installed flag
    echo "📥 Installing requirements with --ignore-installed..."
    pip install --ignore-installed -r requirements.txt
    
    # If that fails, try with no-deps and manual resolution
    if [ $? -ne 0 ]; then
        echo "⚠️ Standard installation failed, trying manual dependency resolution..."
        
        # Install core dependencies first
        echo "📥 Installing core dependencies..."
        pip install --no-deps click==8.1.7
        pip install --no-deps redis==5.0.1
        pip install --no-deps APScheduler==3.10.1
        pip install --no-deps paramiko==3.4.0
        pip install --no-deps fabric==3.1.0
        
        # Then install the rest
        pip install -r requirements.txt --ignore-installed
    fi
    
    echo "✅ Alternative installation complete!"
    echo "Environment: ~/venv/grok-ui-alt"
}

# Run if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
