#!/bin/bash

# Exit on error
set -e

echo "Starting setup..."

# 1. Update and install vim and unzip
echo "Updating system and installing required packages..."
apt update
apt install -y vim unzip nano python3-opencv

# 2. Check CUDA version
echo "CUDA version:"
nvidia-smi

# 3. Install uv
echo "Installing uv..."
pip install uv

# 4. Create virtual environment
echo "Creating virtual environment..."
uv venv

# 5. Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# 6. Install requirements
echo "Installing requirements..."
uv pip install -r requirements.txt

echo "Setup completed successfully!"
echo "Virtual environment is activated and ready to use."
