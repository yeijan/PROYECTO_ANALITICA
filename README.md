# EDA — Predicción de Formaciones SENA

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
├── EDA_Formacion_SENA.ipynb          # Notebook principal con el EDA completo
├── dataset_formacion_sena.csv        # Dataset procesado
├── DATOS/                            # Archivos fuente XML exportados de Sofia Plus
│   ├── DF-04_1.xml
│   ├── DF-08_1.xml
│   ├── DF-09_1.xml
│   ├── DF-10_1.xml
│   ├── DF-12_1.xml
│   ├── DF-13_1.xml
│   ├── DF-14_1.xml
│   └── DF-14A_1.xml
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
