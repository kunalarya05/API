FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY classification/requirements-deploy.txt ./
RUN pip install -r requirements-deploy.txt
COPY classification/app.py classification/price_integration.py classification/class_names.json classification/build_deploy.py ./
COPY classification/best_e_waste_model_v5.keras ./
COPY classification/e-waste-ml-price-engine-new-main/ ./e-waste-ml-price-engine-new-main/
RUN python build_deploy.py
RUN useradd --create-home --uid 10001 apiuser && chown -R apiuser:apiuser /app
USER apiuser
EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "1", "--timeout", "180", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
