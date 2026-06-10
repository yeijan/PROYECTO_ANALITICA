# ── Imagen base ──────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Metadatos
LABEL maintainer="Yeison Acero, Michael Cardenas"
LABEL description="API REST para predicción de deserción estudiantil SENA"
LABEL version="1.0.0"

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CLF_MODEL_PATH=/app/models/mejor_clasificador.pkl \
    REG_MODEL_PATH=/app/models/mejor_regresor.pkl

WORKDIR /app

# Instalar dependencias primero (capa cacheada)
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código y modelos
COPY api/main.py .
COPY api/models/ ./models/

# Puerto de escucha
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Comando de arranque
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
