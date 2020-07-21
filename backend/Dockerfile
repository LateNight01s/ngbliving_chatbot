FROM python:3.8-slim-buster
LABEL maintainer="srofficialsingh@gmail.com"
WORKDIR /backend
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt &&\
    python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')"
COPY . .
ENV PORT=8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
