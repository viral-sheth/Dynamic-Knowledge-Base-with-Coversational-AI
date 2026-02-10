import os
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .test_rag import HealthcareRAG

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
AZURE_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER_NAME", "pdfs")
AZURE_BLOB_PREFIX = os.getenv("AZURE_BLOB_PREFIX", "").strip().strip("/")


def _azure_blob_url(filename: str | None) -> str | None:
    """Build a blob URL from connection string + container, if possible."""
    if not filename or not AZURE_CONN:
        return None
    parts = dict(
        kv.split("=", 1) for kv in AZURE_CONN.split(";") if "=" in kv and kv.strip()
    )
    account = parts.get("AccountName")
    suffix = parts.get("EndpointSuffix", "core.windows.net")
    if not account:
        return None
    prefix = f"{AZURE_BLOB_PREFIX}/" if AZURE_BLOB_PREFIX else ""
    return f"https://{account}.blob.{suffix}/{AZURE_CONTAINER}/{prefix}{filename}"


class ChatRequest(BaseModel):
    question: str


class ChatSource(BaseModel):
    title: str
    url: str = ""
    filename: str | None = None
    page_number: int | None = None
    payer_name: str | None = None
    rule_type: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[ChatSource] = []


app = FastAPI(title="Healthcare Policy Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PDF_STATIC_DIR = Path("payer_pdfs")
PDF_STATIC_DIR.mkdir(exist_ok=True)
app.mount("/pdfs", StaticFiles(directory=str(PDF_STATIC_DIR)), name="pdfs")


def _init_rag() -> HealthcareRAG:
    json_path = Path("healthcare_rules_export.json")
    if not json_path.exists():
        raise FileNotFoundError(
            "Required file 'healthcare_rules_export.json' was not found in the project root."
        )
    return HealthcareRAG(str(json_path))


rag = _init_rag()


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    # Simple payer/state heuristic
    q_lower = question.lower()
    payer_filter = None
    state_filter = None

    if "united" in q_lower:
        # Prefer Community Plan if user mentions United broadly
        payer_filter = {"payer_name": "UnitedHealthcare Community Plan"}
    elif "countycare" in q_lower or "county care" in q_lower:
        payer_filter = {"payer_name": "CountyCare Health Plan"}
    elif "cigna" in q_lower:
        payer_filter = {"payer_name": "Cigna"}
    elif "florida blue" in q_lower:
        payer_filter = {"payer_name": "Florida Blue"}

    if "arizona" in q_lower:
        state_filter = {"geographic_scope": "Arizona"}
    elif "florida" in q_lower:
        state_filter = {"geographic_scope": "Florida"}
    elif "louisiana" in q_lower:
        state_filter = {"geographic_scope": "Louisiana"}
    elif "indiana" in q_lower:
        state_filter = {"geographic_scope": "Indiana"}

    filter_by = None
    if payer_filter and state_filter:
        filter_by = {**payer_filter, **state_filter}
    elif payer_filter:
        filter_by = payer_filter
    elif state_filter:
        filter_by = state_filter

    results = rag.search(question, top_k=3, filter_by=filter_by)
    if not results and filter_by:
        # Fallback to unfiltered search if filter was too strict
        results = rag.search(question, top_k=3, filter_by=None)
    retrieval_answer = rag.format_answer(question, results)

    use_llm = os.getenv("ENABLE_LLM", "false").lower() == "true"
    final_answer = retrieval_answer
    if use_llm:
        llm_answer = rag.generate_llm_answer(question, results)
        # If LLM is unsure/empty, fall back to retrieval
        if llm_answer:
            lower = llm_answer.lower()
            if "not sure" in lower or "insufficient" in lower:
                final_answer = retrieval_answer
            else:
                final_answer = llm_answer
        else:
            final_answer = retrieval_answer

    sources: List[Dict] = []
    for r in results:
        rule = r["rule"]
        title_parts = [rule.get("payer_name") or "", rule.get("rule_title") or ""]
        title = " · ".join([p for p in title_parts if p]).strip() or "Policy rule"
        filename = rule.get("filename")
        url = (
            rule.get("source_url")
            or rule.get("pdf_url")
            or rule.get("provider_portal_url")
            or _azure_blob_url(filename)
            or (f"{API_BASE_URL}/pdfs/{filename}" if filename else "")
        )
        sources.append(
            {
                "title": title,
                "url": url,
                "filename": filename,
                "page_number": rule.get("page_number"),
                "payer_name": rule.get("payer_name"),
                "rule_type": rule.get("rule_type"),
            }
        )

    return ChatResponse(answer=final_answer, sources=[ChatSource(**s) for s in sources])


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}
