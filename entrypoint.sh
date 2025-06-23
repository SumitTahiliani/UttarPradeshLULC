#!/bin/bash

echo "Running Streamlit app"
echo "PORT environment variable: '$PORT'"

PORT_TO_USE=${PORT:-8501}
echo "Using port: $PORT_TO_USE"

streamlit run app.py --server.port=$PORT_TO_USE --server.address=0.0.0.0