# GPU-enabled Ubuntu 22.04 (CUDA 12.3)
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Base tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git curl ca-certificates build-essential \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    byobu \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl --insecure -LsSf https://astral.sh/uv/install.sh | sh

# Add UV permanently to PATH for all shells
ENV PATH="/root/.cargo/bin:${PATH}"

# Also create a symlink (bulletproof)
RUN ln -s /root/.cargo/bin/uv /usr/local/bin/uv

# Set working directory
WORKDIR /workspace

# Use bash
SHELL ["/bin/bash", "-lc"]

# Debug info
RUN python3 --version && pip3 --version && uv --version
