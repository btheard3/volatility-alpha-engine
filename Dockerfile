FROM python:3.11-slim

# Workdir inside the container
WORKDIR /app

# System deps (if you already had these, keep them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Let Cloud Run tell us which port to use; default to 8501 for local dev
ENV PORT=8501

# IMPORTANT: bind Streamlit to $PORT and 0.0.0.0
CMD ["bash", "-c", "streamlit run dashboards/option_screener_v0/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0"]

