#!/bin/bash

echo "Running Streamlit app"
echo "PORT environment variable: '$PORT'"

# Use PORT if set, otherwise default to 8501
PORT_TO_USE=${PORT:-8501}
echo "Using port: $PORT_TO_USE"

streamlit run app.py --server.port=$PORT_TO_USE --server.address=0.0.0.0