# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Deployment platforms provide PORT, default for local testing
ENV PORT 8080

# Set the working directory in the container
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv==0.4.30

# Copy the requirements files
COPY pyproject.toml uv.lock* ./

# Use uv pip install into the system environment, allow dependencies
RUN uv pip install --system --no-cache -r pyproject.toml

# Pre-download and cache the embedding model
# This ensures the model is part of the image and doesn't need to be downloaded on container start.
# It will be cached to the default location (e.g., /root/.cache/torch/sentence_transformers or similar)
RUN python -c "from sentence_transformers import SentenceTransformer; print('Downloading and caching model...'); SentenceTransformer('multi-qa-mpnet-base-dot-v1'); print('Model cached.')"

# Copy the rest of the application code into the container
# Exclude storage here if using volumes, include if not.
COPY . .

# --- CRITICAL: Copy the storage directory into the image --- 
# This is only needed if NOT using Fly.io volumes + manual data upload.
# If using volumes, comment this line out.
COPY storage ./storage

# Inform Docker that the container listens on the port specified by $PORT
EXPOSE $PORT

# Run the actual application using uvicorn (shell form for $PORT expansion)
CMD uvicorn mcp_server.app:app --host 0.0.0.0 --port $PORT
# CMD for debugging: Run the minimal test_asgi:application
# CMD uvicorn test_asgi:application --host 0.0.0.0 --port $PORT
# CMD python -V && echo "PORT is $PORT" 