FROM ghcr.io/astral-sh/uv:debian-slim

SHELL ["/bin/bash", "-c"]

WORKDIR /app

COPY requirements.txt .

# Create a virtual environment and install dependencies
RUN uv python install 3.12 && \
    uv venv .venv && \
    uv pip install --no-cache-dir -r requirements.txt && \
    uv pip uninstall pinecone-plugin-inference

# Copy application code
COPY . .

# copy /data folder
COPY data /app/data

# Expose the port the app runs on
EXPOSE 8080

# Run the application - using shell form for environment variable expansion
CMD sh -c "uv run streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --theme.base=light --server.headless=true"