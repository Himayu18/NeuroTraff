# Step 1: Use a stable lightweight base
FROM python:3.12-slim

# Step 2: Prevent Python from writing .pyc files & buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Step 3: Set working directory
WORKDIR /app

# Step 4: Install system dependencies for ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Step 5: Copy requirements first for caching
COPY requirements.txt .

# Step 6: Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 7: Copy your project
COPY . .

# Step 8: Expose Flask port
EXPOSE 5000

# Step 9: Run with Gunicorn
CMD ["gunicorn", "app:app", "-w", "2", "--timeout", "300", "-b", "0.0.0.0:5000"]
