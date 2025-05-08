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
COPY pyproject.toml uv.lock ./

# Install project dependencies using uv
RUN uv sync --no-dev

# Copy the rest of the application code into the container
# Exclude storage here if using volumes, include if not.
COPY . .

# --- CRITICAL: Copy the storage directory into the image --- 
# This is only needed if NOT using Fly.io volumes + manual data upload.
# If using volumes, comment this line out.
# COPY storage ./storage

# Inform Docker that the container listens on the port specified by $PORT
EXPOSE $PORT

# Run the application using uv run
CMD ["uv", "run", "python", "-m", "mcp_server.main"] 