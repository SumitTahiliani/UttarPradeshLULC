#!/bin/bash

# Print the port for debugging
echo "Starting Streamlit on port $PORT"

# Run Streamlit on the dynamic port
exec streamlit run app.py \
    --server.port=$PORT \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
