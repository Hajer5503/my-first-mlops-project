FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py train.py save_dataset.py .
COPY data/iris.csv data/

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Run the app
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]