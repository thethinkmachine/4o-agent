FROM ghcr.io/astral-sh/uv:0.6.0-python3.12-bookworm-slim

# Update and install required packages including Node.js and npm
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    nodejs \
    npm \
    tree \
    && npm install prettier@3.4.2 --global \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN uv pip install -r requirements.txt --system

COPY . .
EXPOSE 8000
RUN mkdir -p /app/temp && chmod 777 /app/temp
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
