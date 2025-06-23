FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libexpat1 \
    libgdal32 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN echo '#!/bin/bash\necho "Running Streamlit app"\nstreamlit run app.py --server.port=$PORT --server.address=0.0.0.0' > entrypoint.sh
RUN chmod +x entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/bin/bash"]
CMD ["entrypoint.sh"]