# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for snscrape (requests, lxml), geopandas removed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code
COPY app ./app
COPY .streamlit ./.streamlit

# Expose Streamlit port
EXPOSE 8501

# Healthcheck: Streamlit live ping
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "app/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
