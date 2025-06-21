FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data

# Expose ports for both Streamlit and FastAPI
EXPOSE 8501 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting FabAgent..."\n\
echo "Available services:"\n\
echo "  - Streamlit App: http://localhost:8501"\n\
echo "  - FastAPI Backend: http://localhost:8000"\n\
echo "  - API Documentation: http://localhost:8000/docs"\n\
echo ""\n\
echo "Starting FastAPI backend..."\n\
python api.py &\n\
echo "Starting Streamlit app..."\n\
streamlit run app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"] 