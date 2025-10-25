# Multi-stage Dockerfile for Drug Discovery Assistant
# Optimized for both GPU and CPU deployment with size optimization

# ============================================================================
# Stage 1: Base image with CUDA support (GPU variant)
# ============================================================================
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as gpu-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH="/opt/miniconda3/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    software-properties-common \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/miniconda3 \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

# ============================================================================
# Stage 2: CPU-only base image
# ============================================================================
FROM python:3.9-slim-bullseye as cpu-base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    software-properties-common \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Stage 3: Application builder (GPU variant)
# ============================================================================
FROM gpu-base as gpu-builder

WORKDIR /app

# Copy environment configuration
COPY environment.yml requirements.txt ./

# Create conda environment
RUN conda env create -f environment.yml \
    && conda clean -afy \
    && find /opt/miniconda3 -type d -name __pycache__ -exec rm -rf {} + \
    && find /opt/miniconda3 -type f -name "*.py[co]" -delete

# Activate environment and install additional requirements
RUN echo "source activate drug-discovery" > ~/.bashrc
ENV PATH="/opt/miniconda3/envs/drug-discovery/bin:$PATH"

# Install additional pip dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 4: Application builder (CPU variant) 
# ============================================================================
FROM cpu-base as cpu-builder

WORKDIR /app

# Copy requirements
COPY requirements.txt ./

# Create requirements for CPU-only installation
RUN sed 's/torch==2.0.1+cu118/torch==2.0.1+cpu/g' requirements.txt > requirements-cpu.txt \
    && sed -i 's/torchvision==0.15.2+cu118/torchvision==0.15.2+cpu/g' requirements-cpu.txt \
    && sed -i 's/torchaudio==2.0.2+cu118/torchaudio==2.0.2+cpu/g' requirements-cpu.txt \
    && sed -i '/--extra-index-url.*cu118/d' requirements-cpu.txt \
    && echo "--extra-index-url https://download.pytorch.org/whl/cpu" >> requirements-cpu.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-cpu.txt

# ============================================================================
# Stage 5: Final GPU image
# ============================================================================
FROM gpu-base as gpu-final

WORKDIR /app

# Copy conda environment from builder
COPY --from=gpu-builder /opt/miniconda3 /opt/miniconda3

# Set up environment
ENV PATH="/opt/miniconda3/envs/drug-discovery/bin:$PATH"
ENV CONDA_DEFAULT_ENV=drug-discovery

# Copy application code
COPY . .

# Set up directories
RUN mkdir -p /app/data/models /app/data/processed /app/logs \
    && chmod -R 755 /app/scripts

# GPU memory optimization for GTX 1650
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.6"
ENV CUDA_LAUNCH_BLOCKING="1"
ENV TF_FORCE_GPU_ALLOW_GROWTH="true"

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

# ============================================================================
# Stage 6: Final CPU image
# ============================================================================
FROM cpu-base as cpu-final

WORKDIR /app

# Copy Python environment from builder
COPY --from=cpu-builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=cpu-builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Set up directories
RUN mkdir -p /app/data/models /app/data/processed /app/logs \
    && chmod -R 755 /app/scripts

# CPU optimization settings
ENV OMP_NUM_THREADS="4"
ENV MKL_NUM_THREADS="4"
ENV FORCE_CPU="1"

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

# ============================================================================
# Final stage selection based on build argument
# ============================================================================
ARG BUILD_TYPE=gpu
FROM ${BUILD_TYPE}-final as final

# Labels for metadata
LABEL maintainer="Drug Discovery Assistant Team"
LABEL description="AI-powered drug discovery platform optimized for GTX 1650"
LABEL version="1.0.0"
LABEL gpu_support="${BUILD_TYPE}"

# Create non-root user for security
RUN groupadd -r drugdiscovery && useradd -r -g drugdiscovery -s /bin/bash drugdiscovery \
    && chown -R drugdiscovery:drugdiscovery /app

USER drugdiscovery

# Final working directory
WORKDIR /app

# Default entrypoint with configuration
ENTRYPOINT ["python", "-m", "streamlit", "run", "app/main.py"]

# Default arguments (can be overridden)
CMD ["--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.maxUploadSize=200"]