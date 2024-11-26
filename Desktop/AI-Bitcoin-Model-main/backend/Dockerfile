# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the backend folder into the container
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port Flask runs on (e.g., 5000)
EXPOSE 5001

# Run the Flask application
CMD ["python", "main.py"]
