# Use NVIDIA CUDA base image for T4 GPU support
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA support for T4 GPU
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Clone the CHF repository
RUN git clone https://github.com/selayercili/CHF.git .

# Install requirements.txt from the repository
RUN pip install -r requirements.txt

# Install your specific additional packages
RUN pip install plotly kaleido colorlog CoolProp imbalanced-learn

# Install core data science packages
RUN pip install pandas numpy matplotlib seaborn

# Create necessary directories
RUN mkdir -p data logs results

# Default command - keep container running
CMD ["tail", "-f", "/dev/null"]
