# Use Python 3.11 for better memory management and speed
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies including scikit-learn and xgboost
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files (main.py, model.json, mapping.json)
COPY . .

# Run the application using the python command
# This allows main.py to handle the PORT environment variable dynamically
CMD ["python", "main.py"]