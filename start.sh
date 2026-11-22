#!/bin/bash
set -e

# Start uvicorn backend on 127.0.0.1:8000 in the background
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 &

# Wait up to 2 minutes until GET /health answers
echo "Waiting for backend to be ready..."
for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1 || python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" > /dev/null 2>&1; then
        echo "Backend is ready."
        break
    fi
    sleep 1
    if [ "$i" -eq 120 ]; then
        echo "Backend did not start in time."
        exit 1
    fi
done

# Start Streamlit
exec streamlit run frontend/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableXsrfProtection false \
    --server.enableCORS false
