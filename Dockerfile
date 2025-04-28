# Use an official lightweight Python image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy all project files to container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install fastapi uvicorn torch transformers scikit-learn

# Expose the port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

