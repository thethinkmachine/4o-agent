FROM python:3.12-slim-bookworm

# Set the working directory inside the container
WORKDIR /app

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates

# Download and run a script from a remote URL
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Update PATH environment variable
ENV PATH="/root/.local/bin/:$PATH"

# Copy all files from the current directory to the container
COPY . /app/

# Install Python dependencies
RUN ["uv", "pip", "install", "-r", "requirements.txt", "--system"]

# Expose port 8000
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "run", "app.py"]
