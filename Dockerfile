FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the src directory and other necessary files
COPY src ./src

# Copy the .env file to the root directory
COPY .env .

# Expose the port that Gunicorn will run on
EXPOSE 42069

# Run Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:42069", "src.server:app"]
