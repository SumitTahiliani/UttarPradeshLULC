FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    g++ \
    libgdal-dev \
    gdal-bin \
    libexpat1 \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

ENTRYPOINT exec streamlit run app.py --server.port=${PORT} --server.enableCORS=false --server.enableXsrfProtection=false
