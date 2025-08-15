#!/bin/bash
# Comprehensive troubleshooting and installation guide
# Run this when other methods fail

echo "🔧 Grok UI Installation Troubleshooter"
echo "====================================="

# Function to check Python version
check_python() {
    echo "🐍 Checking Python installation..."
    python3 --version
    which python3
    
    # Check if pip is available
    if command -v pip3 &> /dev/null; then
        echo "✅ pip3 available"
        pip3 --version
    else
        echo "❌ pip3 not found"
        return 1
    fi
}

# Function to check virtual environment
check_venv() {
    echo "📦 Checking virtual environment..."
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo "✅ Virtual environment active: $VIRTUAL_ENV"
        return 0
    else
        echo "⚠️ No virtual environment active"
        return 1
    fi
}

# Function to diagnose dependency conflicts
diagnose_conflicts() {
    echo "🔍 Diagnosing dependency conflicts..."
    
    # Check for conflicting packages
    echo "Checking for known problematic packages..."
    pip list | grep -E "(click|redis|paramiko|fabric|apscheduler)" || echo "No conflicting packages found"
    
    # Check pip freeze for exact versions
    echo "Current package versions:"
    pip freeze | grep -E "(click|redis|paramiko|fabric|APScheduler)" || echo "Packages not installed"
}

# Function to suggest solutions
suggest_solutions() {
    echo "💡 Suggested Solutions:"
    echo "1. System Dependencies:"
    echo "   sudo apt install python3-dev libffi-dev libssl-dev build-essential"
    echo ""
    echo "2. Clean Installation:"
    echo "   rm -rf ~/venv/grok-ui*"
    echo "   python3 -m venv ~/venv/grok-ui-clean"
    echo "   source ~/venv/grok-ui-clean/bin/activate"
    echo "   pip install --upgrade pip"
    echo "   pip install -r requirements.txt --no-cache-dir"
    echo ""
    echo "3. Individual Package Installation:"
    echo "   pip install click==8.1.7"
    echo "   pip install redis==5.0.1"
    echo "   pip install APScheduler==3.10.1"
    echo "   pip install paramiko==3.4.0"
    echo "   pip install fabric==3.1.0"
    echo ""
    echo "4. Docker Alternative:"
    echo "   ./docker_test.sh"
    echo ""
    echo "5. System Package Manager (Ubuntu/Debian):"
    echo "   sudo apt install python3-click python3-redis"
}

# Main troubleshooting function
main() {
    echo "Starting diagnostic..."
    
    # Basic checks
    check_python
    python_ok=$?
    
    check_venv
    venv_ok=$?
    
    # Diagnose current state
    diagnose_conflicts
    
    # Run import test if possible
    if [[ -f "test_imports.py" ]]; then
        echo "🧪 Running import test..."
        python test_imports.py
        test_result=$?
        
        if [[ $test_result -eq 0 ]]; then
            echo "🎉 All tests passed! Installation appears to be working."
            exit 0
        fi
    fi
    
    # Suggest solutions
    suggest_solutions
    
    echo ""
    echo "📋 Summary:"
    echo "  Python: $([ $python_ok -eq 0 ] && echo "✅" || echo "❌")"
    echo "  Virtual Env: $([ $venv_ok -eq 0 ] && echo "✅" || echo "⚠️")"
    echo ""
    echo "Choose an installation method above and try again."
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
