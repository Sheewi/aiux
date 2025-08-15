#!/bin/bash
# Docker-based installation and testing
# Use when local installation is problematic

echo "🐳 Docker-based Installation and Testing"
echo "========================================"

# Create Dockerfile for testing
cat > Dockerfile.test << 'EOF'
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY test_imports.py .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --ignore-installed -r requirements.txt

# Default command
CMD ["python", "test_imports.py"]
EOF

# Build and run Docker container
echo "🔨 Building Docker image..."
docker build -f Dockerfile.test -t grok-ui-test .

echo "🚀 Running tests in Docker container..."
docker run --rm grok-ui-test

# Cleanup
echo "🧹 Cleaning up..."
rm -f Dockerfile.test

echo "✅ Docker testing complete!"
