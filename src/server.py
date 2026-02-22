"""
FastAPI server – /diagnose endpoint
POST {"query": "free text symptoms"} → diagnosis JSON
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from src.symptom_extractor import SymptomExtractor
from src.retriever import MedicalRetriever
from src.explainer import ExplanationGenerator

# ── env / config ───────────────────────────────────────────────────────────────
VLLM_URL    = os.getenv("VLLM_120B_URL", "http://host.docker.internal:11437/v1")
MODEL_NAME  = os.getenv("LLM_MODEL", "gpt-oss-120b")
GRAPH_DIR   = os.getenv("GRAPH_DIR", "./data/graphs_new")
CHROMA_DIR  = os.getenv("CHROMA_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "ai-forever/FRIDA")
DEVICE      = os.getenv("DEVICE", "cpu")
TOP_N       = int(os.getenv("TOP_N", "3"))
API_KEY = os.getenv("API_KEY", "sk-BDVloWBwHCr5oltlXwyhtA")


# ── shared state (loaded at startup) ──────────────────────────────────────────
state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = OpenAI(base_url=VLLM_URL, api_key=API_KEY)
    state["extractor"]  = SymptomExtractor(client, model=MODEL_NAME)
    state["retriever"]  = MedicalRetriever(
        graph_dir=GRAPH_DIR,
        chroma_dir=CHROMA_DIR,
        embed_model=EMBED_MODEL,
        device=DEVICE,
    )
    state["explainer"] = ExplanationGenerator(client, model=MODEL_NAME)
    yield
    state.clear()


app = FastAPI(title="Medical Diagnosis API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── schemas ────────────────────────────────────────────────────────────────────
class DiagnoseRequest(BaseModel):
    symptoms: str


class DiagnosisItem(BaseModel):
    rank:        int
    diagnosis:   str
    icd10_code:  str
    explanation: str


class DiagnoseResponse(BaseModel):
    diagnoses: list[DiagnosisItem]


# ── endpoint ───────────────────────────────────────────────────────────────────
@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    if not request.symptoms.strip():
        # s
        raise HTTPException(status_code=400, detail="query must not be empty")

    t0 = time.perf_counter()

    # 1. Extract symptoms from free text
    symptoms: list[str] = state["extractor"].extract(request.symptoms)
    print(symptoms)
    
    symptoms = [request.symptoms] + symptoms   # fallback: treat whole query as one symptom

    # 2. Retrieve + graph-score candidates
    candidates: list[dict] = state["retriever"].retrieve(symptoms)
    print(candidates)
    # candidates = candidates[:TOP_N]

    if not candidates:
        return DiagnoseResponse(diagnoses=[])

    # 3. Generate LLM explanations
    explanations: dict[str, str] = state["explainer"].generate(symptoms, candidates)

    # 4. Build response
    diagnoses = [
        DiagnosisItem(
            rank       = c["rank"],
            diagnosis  = c["diagnosis"],
            icd10_code = c["icd10_code"],
            explanation= explanations.get(c["uid"], ""),
        )
        for c in candidates
    ]

    elapsed = round(time.perf_counter() - t0, 3)
    print(f"[/diagnose] latency={elapsed}s symptoms={symptoms}")

    return DiagnoseResponse(diagnoses=diagnoses)


@app.get("/health")
async def health():
    return {"status": "ok"}
