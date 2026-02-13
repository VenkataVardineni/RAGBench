FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model (optional but recommended)
RUN python -m spacy download en_core_web_sm || true

# Copy application code
COPY . .

# Install package
RUN pip install -e .

# Set working directory
WORKDIR /app

# Default command
CMD ["python", "-m", "ragbench.eval.run_eval", "--config", "configs/demo.yaml"]

