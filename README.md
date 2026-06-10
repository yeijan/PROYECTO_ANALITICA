# Predicción de Deserción Estudiantil — SENA Centro Amazonas

## Descripción del dataset

| Campo | Detalle |
|---|---|
| **Nombre** | Fichas de Formación — Centro para la Biodiversidad y el Turismo del Amazonas |
| **Fuente** | Sistema Sofia Plus — Reportes DF-14A y DF-08 |
| **Contexto** | Datos de fichas de formación del SENA con el objetivo de predecir la deserción estudiantil por ficha, analizar patrones por trimestre y recomendar estrategias para los niveles Técnico, Tecnólogo y Operario |
| **Dimensiones** | 153 fichas × 38 variables |
| **Tipo de problema** | Clasificación (alta/baja deserción) + Regresión (tasa de deserción continua) |
| **Periodo** | Octubre 2022 — Mayo 2026 |

## Integrantes

- Yeison Acero
- Michael Cardenas

## Estructura del repositorio

```
PROYECTO_ANALITICA/
├── EDA_Formacion_SENA.ipynb          # Parte 1 — EDA completo
├── dataset_formacion_sena.csv        # Dataset procesado (153 fichas × 38 vars)
├── requirements.txt                  # Dependencias del proyecto
├── DATOS/                            # Archivos fuente XML exportados de Sofia Plus
│   ├── DF-04_1.xml
│   ├── DF-08_1.xml … DF-14A_1.xml
│
├── parte2/
│   └── 01_modeling_pipeline.ipynb    # Parte 2 — Pipeline MLflow + 6 modelos
│
├── api/                              # Despliegue REST
│   ├── main.py                       # FastAPI app
│   ├── requirements.txt              # Dependencias de la API
│   └── models/                       # Modelos exportados (generados por el notebook)
│       ├── mejor_clasificador.pkl
│       ├── mejor_regresor.pkl
│       └── model_metadata.json
│
├── Dockerfile                        # Imagen Docker de la API
├── docker-compose.yml                # API + MLflow UI
├── mlruns/                           # Artefactos MLflow (generados por el notebook)
└── docs/
    └── Documento_Explicacion_Celdas_EDA_Formacion_SENA.md
```

## Contenido del EDA

1. Inspección inicial y estadísticas descriptivas
2. Análisis de valores faltantes y tratamiento justificado
3. Distribución de la variable objetivo (`TASA_DESERCION`)
4. Análisis univariado — variables numéricas y categóricas con detección de outliers
5. Análisis bivariado — correlaciones, scatterplots y boxplots
6. Preparación para el modelado — features seleccionadas y variables redundantes
7. Hallazgos y conclusiones

## Principales hallazgos

- **Variable con mayor correlación con el target:** `AÑO_APERTURA` (r = −0.64) — la deserción ha disminuido con el tiempo
- **Desbalance de clases:** leve — 66% baja deserción / 34% alta deserción
- **Variable redundante:** `DURACION_ETAPA_LECTIVA` (r = 0.98 con `DURACION_MAXIMA`)
- **Data leakage a excluir:** `TOTAL_DESERCION`, `CERTIFICADO`, `CANCELADO`, `RETIRO_VOLUNTARIO`
- **Features seleccionadas para modelado:** `NIVEL_FORMACION`, `JORNADA`, `TRIMESTRE_APERTURA`, `AÑO_APERTURA`, `CUPO`, `DURACION_MAXIMA`, `OCUPACION_CUPO`, `PROGRAMA_ESPECIAL`

---

## Parte 2 — Pipeline de Modelado y Despliegue

### Modelos entrenados

| Tarea | Modelo | Optimización |
|---|---|---|
| Clasificación | Logistic Regression (baseline) | GridSearchCV |
| Clasificación | Random Forest Classifier | RandomizedSearchCV |
| Clasificación | XGBoost Classifier | RandomizedSearchCV |
| Regresión | Ridge Regression (baseline) | GridSearchCV |
| Regresión | Random Forest Regressor | RandomizedSearchCV |
| Regresión | XGBoost Regressor | RandomizedSearchCV |

**Métrica clave clasificación:** Recall clase Alta — minimizar falsos negativos (fichas en riesgo que no son detectadas).

### Cómo ejecutar

**1. Instalar dependencias**
```bash
pip install -r requirements.txt
```

**2. Entrenar modelos y registrar experimentos**
```bash
# Ejecutar el notebook completo
jupyter notebook parte2/01_modeling_pipeline.ipynb
```

**3. Ver experimentos en MLflow UI**
```bash
mlflow ui --backend-store-uri mlruns
# Abrir http://localhost:5000
```

**4. Levantar la API con Docker**
```bash
# Construir y arrancar
docker compose up --build

# API disponible en http://localhost:8000/docs
# MLflow UI en http://localhost:5000
```

**5. Probar la API**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "NIVEL_FORMACION": "TÉCNICO",
    "JORNADA": "NOCTURNA",
    "TRIMESTRE_APERTURA": 3,
    "AÑO_APERTURA": 2024,
    "CUPO": 30,
    "DURACION_MAXIMA": 2112,
    "OCUPACION_CUPO": 0.85,
    "PROGRAMA_ESPECIAL": "NO"
  }'
```

### Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio |
| GET | `/models/info` | Info de los modelos cargados |
| POST | `/predict` | Predicción individual |
| POST | `/predict/batch` | Predicción en lote (máx. 500 fichas) |
| GET | `/docs` | Documentación interactiva (Swagger UI) |
