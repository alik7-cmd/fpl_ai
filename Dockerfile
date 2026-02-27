# Use official Python base image
FROM python:3.10-slim

# Set work directory inside container
WORKDIR /app

# Install system dependencies (including GLPK solver for PuLP)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    glpk-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
