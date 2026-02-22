# Medical Diagnosis Assistant – QazCode 2026

## Architecture

```
User query (free text)
        │
        ▼
┌─────────────────────┐
│  SymptomExtractor   │  LLM (GPT-OSS) → JSON list of symptom strings
└────────┬────────────┘
         │ symptom strings
         ▼
┌─────────────────────┐
│   VectorIndex       │  ChromaDB + ruRoberta (last hidden layer pooling)
│   (ChromaDB)        │  → top-K matched symptom node UIDs
└────────┬────────────┘
         │ matched node UIDs
         ▼
┌─────────────────────┐
│   GraphScorer       │  NetworkX DiGraph
│   (NetworkX)        │
│                     │  score(D) = Σ direct_supports
│                     │           − Σ direct_contradicts × 1.5
│                     │           + Σ indirect co-occurs  × 0.4
│                     │           + Σ appear-with boost   × 0.3
└────────┬────────────┘
         │ top-N ranked diagnoses
         ▼
┌─────────────────────┐
│ ExplanationGenerator│  LLM (GPT-OSS) → clinical explanation per diagnosis
└────────┬────────────┘
         │
         ▼
   /diagnose JSON response
```

## Quick start

```bash
# 1. Build ChromaDB index from extracted graphs (once)
uv run python build_index.py --graph_dir ./data/graphs_new --chroma_dir ./chroma_db

# 2. Start API (port 8080) + Streamlit UI (port 8501)
VLLM_URL=http://<your-vllm-host>/v1 uv run uvicorn src.server:app --port 8080
uv run streamlit run src/app.py
```

## Docker

```bash
docker build -t submission .
docker run -p 8080:8080 -p 8501:8501 \
  -e VLLM_URL=http://<vllm-host>/v1 \
  submission
```

## Environment variables

| Variable     | Default                        | Description                    |
|-------------|--------------------------------|--------------------------------|
| `VLLM_URL`  | `http://localhost:8080/v1`    | GPT-OSS endpoint               |
| `LLM_MODEL` | `gpt-oss-120b`                | Model name                     |
| `GRAPH_DIR` | `./data/graphs_new`           | Extracted graph JSON directory |
| `CHROMA_DIR`| `./chroma_db`                 | ChromaDB persistence path      |
| `EMBED_MODEL`| `ai-forever/ruRoberta-large` | Embedding model                |
| `DEVICE`    | `cpu`                         | `cpu` or `cuda`                |
| `TOP_N`     | `3`                           | Number of diagnoses to return  |

## API

**POST /diagnose**
```json
// Request
{"query": "высокая температура, кашель, одышка"}

// Response
{
  "diagnoses": [
    {
      "rank": 1,
      "diagnosis": "Острый бронхит",
      "icd10_code": "J20.9",
      "explanation": "..."
    }
  ]
}
```
