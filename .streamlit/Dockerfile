FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libexpat1-dev \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV GDAL_CONFIG=/usr/bin/gdal-config

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD sh -c "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"