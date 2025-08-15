#!/usr/bin/env python3
"""
Setup script for Microagents Specialization Matrix.
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="microagents-specialization-matrix",
    version="1.0.0",
    author="Microagents Team",
    author_email="contact@microagents.dev",
    description="Specialized microagents for web automation, data extraction, computer vision, system governance, and API orchestration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microagents/specialization-matrix",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: Browsers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "microagents-demo=examples.demo_all_agents:main",
            "microagents-web=examples.web_automation_examples:main",
            "microagents-api=examples.api_orchestration_examples:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "microagents",
        "automation",
        "web-scraping", 
        "computer-vision",
        "api-orchestration",
        "system-monitoring",
        "playwright",
        "scrapy",
        "opencv",
        "kubernetes",
        "graphql",
        "httpx",
        "onnx"
    ],
    project_urls={
        "Bug Reports": "https://github.com/microagents/specialization-matrix/issues",
        "Source": "https://github.com/microagents/specialization-matrix",
        "Documentation": "https://microagents.readthedocs.io/",
    },
)
