# Backend Dockerfile for Bacterial Simulation FastAPI Application
# Multi-stage build for optimal image size and security

# Base image with Python
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VENV_IN_PROJECT=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1001 fastapi && \
    useradd --uid 1001 --gid fastapi --shell /bin/bash --create-home fastapi

# Set work directory
WORKDIR /app

# Dependencies stage
FROM base AS deps

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base AS runner

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/simulation_results /app/simulation_states /app/simulation_backups && \
    chown -R fastapi:fastapi /app

# Switch to non-root user
USER fastapi

# Copy application code
COPY --chown=fastapi:fastapi . .

# Create startup script for better process management
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting FastAPI server..."\n\
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --worker-class uvicorn.workers.UvicornWorker\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["/app/start.sh"] 