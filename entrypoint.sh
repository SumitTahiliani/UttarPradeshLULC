#!/bin/bash

echo "Running Streamlit app"
echo "PORT environment variable: '$PORT'"

PORT_TO_USE=${PORT:-8501}
echo "Using port: $PORT_TO_USE"

if ! [[ "$PORT_TO_USE" =~ ^[0-9]+$ ]]; then
  echo "Error: PORT '$PORT_TO_USE' is not a valid integer."
  exit 1
fi

streamlit run app.py --server.port=$PORT_TO_USE --server.address=0.0.0.0
