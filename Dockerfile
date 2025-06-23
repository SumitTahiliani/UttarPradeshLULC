FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libexpat1 \
    libgdal32 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN echo '#!/bin/bash\necho "Running Streamlit app"\necho "PORT is set to: $PORT"\n# Use PORT if set, otherwise default to 8501\nPORT_TO_USE=${PORT:-8501}\necho "Using port: $PORT_TO_USE"\nstreamlit run app.py --server.port=$PORT_TO_USE --server.address=0.0.0.0' > entrypoint.sh
RUN chmod +x entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/bin/bash"]
CMD ["entrypoint.sh"]