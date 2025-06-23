FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libexpat1 \
    libexpat1-dev \
    libgdal32 \
    libgdal-dev \
    libproj25 \
    libproj-dev \
    libgeos-c1v5 \
    libgeos-dev \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

ENV GDAL_CONFIG=/usr/bin/gdal-config

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --no-binary rasterio,GDAL rasterio GDAL
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0