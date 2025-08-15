#!/bin/bash
# Step-by-step manual installation
# Use when automated scripts fail

echo "🔧 Manual Step-by-Step Installation"
echo "==================================="

# Step 1: Check environment
echo "Step 1: Checking environment..."
echo "Python version: $(python3 --version)"
echo "Pip version: $(pip --version 2>/dev/null || echo 'pip not found')"
echo "Virtual env: ${VIRTUAL_ENV:-'Not active'}"

# Step 2: Create fresh environment
echo -e "\nStep 2: Creating fresh virtual environment..."
read -p "Create new virtual environment? (y/n): " create_venv
if [[ $create_venv == "y" ]]; then
    rm -rf ~/venv/grok-manual
    python3 -m venv ~/venv/grok-manual
    source ~/venv/grok-manual/bin/activate
    echo "✅ Created and activated: ~/venv/grok-manual"
fi

# Step 3: Upgrade pip
echo -e "\nStep 3: Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Step 4: Install packages one by one
echo -e "\nStep 4: Installing packages individually..."

packages=(
    "click==8.1.7"
    "redis==5.0.1" 
    "requests"
    "pandas"
    "numpy"
    "fastapi"
    "pydantic"
)

failed_packages=()

for package in "${packages[@]}"; do
    echo "Installing $package..."
    if pip install "$package" --no-cache-dir; then
        echo "✅ $package installed"
    else
        echo "❌ $package failed"
        failed_packages+=("$package")
    fi
    echo "---"
done

# Step 5: Test imports
echo -e "\nStep 5: Testing imports..."
python -c "
packages = ['click', 'redis', 'requests', 'pandas', 'numpy', 'fastapi', 'pydantic']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')
"

# Summary
echo -e "\n" + "=" * 40
if [ ${#failed_packages[@]} -eq 0 ]; then
    echo "🎉 Manual installation completed successfully!"
else
    echo "⚠️ Some packages failed:"
    printf '%s\n' "${failed_packages[@]}"
    echo ""
    echo "Try these alternatives:"
    echo "1. pip install --user <package>"
    echo "2. conda install <package>"
    echo "3. apt install python3-<package>"
fi

echo ""
echo "Next steps:"
echo "1. Run: python test_imports.py"
echo "2. If issues persist: ./troubleshoot.sh"
