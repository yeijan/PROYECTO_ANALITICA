"""
API REST — Predicción de Deserción Estudiantil SENA
FastAPI + joblib models entrenados con MLflow
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── Rutas de modelos ────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

CLF_MODEL_PATH  = Path(os.getenv("CLF_MODEL_PATH",  str(MODELS_DIR / "mejor_clasificador.pkl")))
REG_MODEL_PATH  = Path(os.getenv("REG_MODEL_PATH",  str(MODELS_DIR / "mejor_regresor.pkl")))
META_PATH       = MODELS_DIR / "model_metadata.json"

# ── Estado global de modelos ────────────────────────────────────────────────
_clf_model = None
_reg_model = None
_model_meta: dict = {}


def _load_models() -> None:
    global _clf_model, _reg_model, _model_meta
    if not CLF_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Clasificador no encontrado en {CLF_MODEL_PATH}. "
            "Ejecuta el notebook parte2/01_modeling_pipeline.ipynb primero."
        )
    if not REG_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Regresor no encontrado en {REG_MODEL_PATH}. "
            "Ejecuta el notebook parte2/01_modeling_pipeline.ipynb primero."
        )
    _clf_model = joblib.load(CLF_MODEL_PATH)
    _reg_model = joblib.load(REG_MODEL_PATH)
    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as f:
            _model_meta = json.load(f)


# ── Aplicación FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title="SENA Deserción Predictor",
    description=(
        "API REST para predecir la deserción estudiantil en fichas de formación SENA. "
        "Ofrece clasificación binaria (Alta/Baja deserción) y estimación continua de la tasa."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    _load_models()


# ── Schemas Pydantic ─────────────────────────────────────────────────────────

NIVEL_FORMACION_VALUES = ["OPERARIO", "TÉCNICO", "TECNÓLOGO", "TECNICO", "TECNOLOGO"]
JORNADA_VALUES         = ["DIURNA", "NOCTURNA", "MADRUGADA", "MIXTA", "FINES DE SEMANA"]
PROGRAMA_ESPECIAL_VALUES = ["SI", "NO"]


class FichaInput(BaseModel):
    """Datos de una ficha de formación SENA para predicción."""

    NIVEL_FORMACION: str = Field(
        ...,
        description="Nivel de formación: OPERARIO, TÉCNICO o TECNÓLOGO",
        examples=["TÉCNICO"],
    )
    JORNADA: str = Field(
        ...,
        description="Jornada de la ficha",
        examples=["NOCTURNA"],
    )
    TRIMESTRE_APERTURA: int = Field(
        ...,
        ge=1,
        le=4,
        description="Trimestre de apertura (1–4)",
        examples=[3],
    )
    ANNO_APERTURA: int = Field(
        ...,
        ge=2015,
        le=2030,
        description="Año de apertura de la ficha",
        examples=[2024],
        alias="AÑO_APERTURA",
    )
    CUPO: int = Field(
        ...,
        ge=1,
        le=100,
        description="Cupo total de la ficha",
        examples=[30],
    )
    DURACION_MAXIMA: int = Field(
        ...,
        ge=300,
        le=10000,
        description="Duración máxima en horas",
        examples=[2112],
    )
    OCUPACION_CUPO: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Tasa de ocupación del cupo (0.0 – 1.0)",
        examples=[0.85],
    )
    PROGRAMA_ESPECIAL: str = Field(
        ...,
        description="¿Es programa especial? (SI / NO)",
        examples=["NO"],
    )

    model_config = {"populate_by_name": True}

    @field_validator("NIVEL_FORMACION")
    @classmethod
    def validate_nivel(cls, v: str) -> str:
        v_upper = v.upper().strip()
        if v_upper not in [x.upper() for x in NIVEL_FORMACION_VALUES]:
            raise ValueError(
                f"NIVEL_FORMACION debe ser uno de: {NIVEL_FORMACION_VALUES}"
            )
        return v_upper

    @field_validator("JORNADA")
    @classmethod
    def validate_jornada(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("PROGRAMA_ESPECIAL")
    @classmethod
    def validate_programa(cls, v: str) -> str:
        v_upper = v.upper().strip()
        if v_upper not in PROGRAMA_ESPECIAL_VALUES:
            raise ValueError(f"PROGRAMA_ESPECIAL debe ser SI o NO")
        return v_upper


class PrediccionOutput(BaseModel):
    """Resultado de predicción para una ficha."""

    desercion_alta: bool = Field(description="True si se predice alta deserción (>30%)")
    probabilidad_desercion_alta: float = Field(description="Probabilidad de clase Alta [0–1]")
    tasa_desercion_estimada: float = Field(description="Tasa de deserción estimada [0–1]")
    nivel_riesgo: str = Field(description="BAJO / MEDIO / ALTO según probabilidad")
    modelo_clasificador: str = Field(description="Nombre del modelo de clasificación")
    modelo_regresor: str = Field(description="Nombre del modelo de regresión")


class BatchInput(BaseModel):
    fichas: list[FichaInput]


class BatchOutput(BaseModel):
    total: int
    resultados: list[PrediccionOutput]
    resumen: dict


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ficha_to_df(ficha: FichaInput) -> pd.DataFrame:
    data = {
        "NIVEL_FORMACION"  : ficha.NIVEL_FORMACION,
        "JORNADA"          : ficha.JORNADA,
        "TRIMESTRE_APERTURA": ficha.TRIMESTRE_APERTURA,
        "AÑO_APERTURA"     : ficha.ANNO_APERTURA,
        "CUPO"             : ficha.CUPO,
        "DURACION_MAXIMA"  : ficha.DURACION_MAXIMA,
        "OCUPACION_CUPO"   : ficha.OCUPACION_CUPO,
        "PROGRAMA_ESPECIAL": ficha.PROGRAMA_ESPECIAL,
    }
    return pd.DataFrame([data])


def _nivel_riesgo(prob: float) -> str:
    if prob >= 0.65:
        return "ALTO"
    if prob >= 0.35:
        return "MEDIO"
    return "BAJO"


def _predict_single(ficha: FichaInput) -> PrediccionOutput:
    df = _ficha_to_df(ficha)
    prob  = float(_clf_model.predict_proba(df)[0][1])
    tasa  = float(np.clip(_reg_model.predict(df)[0], 0, 1))
    return PrediccionOutput(
        desercion_alta=prob >= 0.5,
        probabilidad_desercion_alta=round(prob, 4),
        tasa_desercion_estimada=round(tasa, 4),
        nivel_riesgo=_nivel_riesgo(prob),
        modelo_clasificador=_model_meta.get("clasificador", {}).get("nombre", "unknown"),
        modelo_regresor=_model_meta.get("regresor", {}).get("nombre", "unknown"),
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "servicio": "SENA Deserción Predictor",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/predict/batch", "/models/info"],
    }


@app.get("/health", tags=["Info"])
def health():
    return {
        "status": "ok",
        "modelos_cargados": _clf_model is not None and _reg_model is not None,
    }


@app.get("/models/info", tags=["Info"])
def models_info():
    if not _model_meta:
        return {"mensaje": "Metadata no disponible. Ejecuta el notebook primero."}
    return _model_meta


@app.post(
    "/predict",
    response_model=PrediccionOutput,
    tags=["Predicción"],
    summary="Predicción individual",
)
def predict(ficha: FichaInput):
    """Recibe los datos de una ficha y retorna la predicción de deserción."""
    if _clf_model is None or _reg_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelos no cargados. Verifica el startup del servidor.",
        )
    return _predict_single(ficha)


@app.post(
    "/predict/batch",
    response_model=BatchOutput,
    tags=["Predicción"],
    summary="Predicción en lote",
)
def predict_batch(payload: BatchInput):
    """Recibe una lista de fichas y retorna predicciones para todas."""
    if _clf_model is None or _reg_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelos no cargados.",
        )
    if len(payload.fichas) == 0:
        raise HTTPException(status_code=400, detail="La lista de fichas está vacía.")
    if len(payload.fichas) > 500:
        raise HTTPException(status_code=400, detail="Máximo 500 fichas por lote.")

    resultados = [_predict_single(f) for f in payload.fichas]

    alto_riesgo = sum(1 for r in resultados if r.nivel_riesgo == "ALTO")
    medio_riesgo = sum(1 for r in resultados if r.nivel_riesgo == "MEDIO")
    bajo_riesgo = sum(1 for r in resultados if r.nivel_riesgo == "BAJO")

    return BatchOutput(
        total=len(resultados),
        resultados=resultados,
        resumen={
            "alto_riesgo": alto_riesgo,
            "medio_riesgo": medio_riesgo,
            "bajo_riesgo": bajo_riesgo,
            "tasa_promedio_estimada": round(
                sum(r.tasa_desercion_estimada for r in resultados) / len(resultados), 4
            ),
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
