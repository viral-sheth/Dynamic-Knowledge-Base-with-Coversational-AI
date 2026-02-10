#!/usr/bin/env python3
"""
Hybrid healthcare policy extractor:
- Docling for PDF parsing (markdown + tables)
- Rule-based extraction + confidence scoring
- AI fallback via HuggingFace Inference API only when confidence is low
- Uploads JSON to Azure Blob Storage

Config (env or .env):
AZURE_STORAGE_CONNECTION_STRING=...
AZURE_CONTAINER_NAME=pdfs              # PDF source container
AZURE_JSON_CONTAINER=policy-json       # JSON output container
CONFIDENCE_THRESHOLD=0.75              # Rules confidence required to skip AI
HF_TOKEN=...                           # Hugging Face token
HF_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
MAX_RETRIES=2                          # AI retries
AI_TIMEOUT=60                          # seconds
LOG_LEVEL=INFO                         # DEBUG/INFO/WARN/ERROR
MAX_WORKERS=1                          # parallel PDF jobs (keep low on low-spec machines)
"""

import os
import re
import json
import logging
import datetime
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from huggingface_hub import InferenceClient
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.backend_options import PdfBackendOptions


# ---------- Logging ----------
def setup_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("hybrid_extraction.log"), logging.StreamHandler()],
    )


# ---------- Docling ----------
def build_converter() -> DocumentConverter:
    format_options = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=ThreadedPdfPipelineOptions(
                do_ocr=False,
                force_backend_text=True,
                accelerator_options=AcceleratorOptions(device="cpu", num_threads=4),
            ),
            backend_options=PdfBackendOptions(enable_local_fetch=False, enable_remote_fetch=False),
        )
    }
    return DocumentConverter(format_options=format_options)


def parse_pdf(converter: DocumentConverter, pdf_path: Path) -> Tuple[str, List[str], int]:
    doc = converter.convert(str(pdf_path))
    markdown = doc.document.export_to_markdown()
    tables_md = []
    for tbl in doc.document.tables:
        try:
            tables_md.append(tbl.export_to_markdown())
        except Exception:
            continue
    pages = len(doc.document.pages)
    return markdown, tables_md, pages


# ---------- Rule-based extraction ----------
PAYER_PATTERNS = {
    "CountyCare": ["countycare", "county care"],
    "United Healthcare": ["uhc", "united healthcare", "unitedhealthcare"],
    "Cigna": ["cigna"],
    "Anthem": ["anthem", "blue cross blue shield", "bcbs"],
    "Aetna": ["aetna"],
    "Humana": ["humana"],
    "Molina": ["molina"],
    "WellCare": ["wellcare", "well care"],
}

POLICY_TYPES = {
    "claims": ["claims submission", "claim filing", "billing", "reimbursement"],
    "appeals": ["appeal", "grievance", "dispute", "reconsideration"],
    "prior_authorization": ["prior auth", "preauthorization", "pa required", "prior authorization"],
    "timely_filing": ["timely filing", "filing deadline", "submission deadline"],
    "credentialing": ["credentialing", "provider enrollment"],
}

DATE_PATTERNS = [
    r"effective[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    r"effective[:\s]+(\w+\s+\d{1,2},?\s+\d{4})",
    r"expir(?:es|ation)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    r"expir(?:es|ation)[:\s]+(\w+\s+\d{1,2},?\s+\d{4})",
]

FILING_PATTERNS = [
    r"(\d+)\s*(?:calendar\s+)?days?.*timely filing",
    r"timely filing.*?(\d+)\s*(?:calendar\s+)?days?",
    r"submit.*?within\s+(\d+)\s*days?",
    r"(\d+)\s*months?",
]


def detect_payer(text: str) -> str:
    lower = text.lower()
    for payer, patterns in PAYER_PATTERNS.items():
        if any(pat in lower for pat in patterns):
            return payer
    return "Unknown"


def detect_policy_type(text: str) -> str:
    lower = text.lower()
    best = ("general", 0)
    for ptype, patterns in POLICY_TYPES.items():
        count = sum(lower.count(pat) for pat in patterns)
        if count > best[1]:
            best = (ptype, count)
    return best[0]


def extract_dates(text: str) -> Tuple[str, str]:
    eff = None
    exp = None
    for pat in DATE_PATTERNS:
        for match in re.finditer(pat, text, flags=re.IGNORECASE):
            val = match.group(1)
            if "expir" in pat and exp is None:
                exp = val
            elif eff is None:
                eff = val
    return eff, exp


def extract_timely_filing(text: str) -> int:
    lower = text.lower()
    for pat in FILING_PATTERNS:
        m = re.search(pat, lower, flags=re.IGNORECASE | re.DOTALL)
        if m:
            try:
                days = int(m.group(1))
                if "month" in pat and days < 50:  # crude month->days
                    days *= 30
                return days
            except Exception:
                continue
    return None


def detect_prior_auth(text: str) -> bool:
    lower = text.lower()
    if "prior auth" in lower or "prior authorization" in lower or "preauthorization" in lower:
        return True
    if "no prior authorization" in lower or "not require prior authorization" in lower:
        return False
    return None


def extract_title(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[0][:200] if lines else ""


def extract_key_requirements(text: str, max_items: int = 5) -> List[str]:
    reqs = []
    for line in text.splitlines():
        if re.match(r"^[-•*]\s+", line.strip()):
            reqs.append(line.strip().lstrip("-•* ").strip())
    return reqs[:max_items]


def extract_with_rules(text: str, tables: List[str]) -> Dict:
    payer = detect_payer(text)
    policy_type = detect_policy_type(text)
    eff, exp = extract_dates(text)
    timely = extract_timely_filing(text)
    pa = detect_prior_auth(text)
    title = extract_title(text)
    reqs = extract_key_requirements(text)

    return {
        "payer": payer,
        "policy_type": policy_type,
        "title": title,
        "effective_date": eff,
        "expiration_date": exp,
        "summary": None,  # AI will fill
        "key_requirements": reqs,
        "timely_filing_days": timely,
        "prior_auth_required": pa,
        "content": text,
        "tables": tables,
    }


# ---------- Confidence ----------
def calculate_confidence(result: Dict) -> float:
    score = 0.0
    if result.get("payer") and result["payer"] != "Unknown":
        score += 0.25
    if result.get("policy_type") and result["policy_type"] != "general":
        score += 0.20
    if result.get("effective_date"):
        score += 0.15
    if result.get("timely_filing_days"):
        score += 0.15
    if result.get("prior_auth_required") is not None:
        score += 0.10
    if result.get("key_requirements") and len(result["key_requirements"]) >= 3:
        score += 0.15
    return round(score, 2)


# ---------- AI Fallback ----------
def build_ai_prompt(text: str, rule_result: Dict) -> str:
    return textwrap.dedent(f"""
    Extract healthcare policy information from this document.

    Rule extraction (needs validation):
    - Payer: {rule_result.get('payer', 'Unknown')}
    - Policy Type: {rule_result.get('policy_type', 'unknown')}
    - Effective Date: {rule_result.get('effective_date', 'not found')}

    Return ONLY valid JSON:
    {{
      "payer": "exact insurance company name",
      "policy_type": "claims|appeals|prior_authorization|timely_filing|credentialing|general",
      "title": "policy document title",
      "effective_date": "YYYY-MM-DD or null",
      "expiration_date": "YYYY-MM-DD or null",
      "timely_filing_days": number or null,
      "prior_auth_required": true|false|null,
      "summary": "1-2 sentence summary",
      "key_requirements": ["req1", "req2", "req3"]
    }}

    Document text:
    {text[:6000]}
    """).strip()


def extract_with_ai(text: str, rule_result: Dict, client: InferenceClient, model: str, timeout: int, retries: int) -> Dict:
    prompt = build_ai_prompt(text, rule_result)
    for attempt in range(1, retries + 1):
        try:
            resp = client.text_generation(
                prompt=prompt,
                model=model,
                max_new_tokens=800,
                temperature=0.15,
                timeout=timeout,
            )
            cleaned = resp.strip().strip("`")
            return json.loads(cleaned)
        except Exception as e:
            if attempt == retries:
                raise
            logging.warning(f"AI attempt {attempt} failed: {e}, retrying...")
    return {}


# ---------- Hybrid extraction per PDF ----------
def extract_hybrid(pdf_path: Path, payer: str, config: Dict, blob_service: BlobServiceClient, json_container: str) -> bool:
    try:
        converter = build_converter()
        markdown, tables, pages = parse_pdf(converter, pdf_path)

        rule_result = extract_with_rules(markdown, tables)
        rule_result["filename"] = pdf_path.name
        rule_result["payer"] = rule_result.get("payer") if rule_result.get("payer") != "Unknown" else payer
        confidence = calculate_confidence(rule_result)

        use_ai = config["enable_ai"] and confidence < config["threshold"]
        result = rule_result

        if use_ai:
            logging.info(f"⚡ AI fallback for {pdf_path.name} (confidence {confidence})")
            try:
                ai_res = extract_with_ai(
                    markdown,
                    rule_result,
                    config["hf_client"],
                    config["hf_model"],
                    config["ai_timeout"],
                    config["max_retries"],
                )
                result = {**rule_result, **ai_res}
                result["ai_used"] = True
            except Exception as e:
                logging.warning(f"AI failed for {pdf_path.name}: {e}")
                result["ai_used"] = False
                result["ai_error"] = str(e)
        else:
            result["ai_used"] = False

        result.setdefault("metadata", {})
        result["metadata"].update({
            "pages": pages,
            "extraction_method": "rules" if not result["ai_used"] else "hybrid",
            "confidence_score": confidence,
            "extracted_date": datetime.date.today().isoformat(),
        })
        result.setdefault("tables", tables)
        result.setdefault("content", markdown)

        # Upload JSON
        url = upload_json(blob_service, json_container, payer, pdf_path.name, result)
        if url:
            logging.info(f"Uploaded JSON: {url}")
            return True
        return False
    except Exception as e:
        logging.error(f"Failed processing {pdf_path.name}: {e}")
        return False


# ---------- Azure helpers ----------
def ensure_container(blob_service: BlobServiceClient, container: str):
    try:
        blob_service.create_container(container)
    except Exception:
        pass


def upload_json(blob_service: BlobServiceClient, container: str, payer: str, filename: str, data: Dict) -> str:
    try:
        blob_name = f"{payer}/{Path(filename).stem}.json"
        blob_client = blob_service.get_blob_client(container=container, blob=blob_name)
        blob_client.upload_blob(json.dumps(data, indent=2), overwrite=True, metadata={
            "payer_name": payer,
            "source_pdf": filename,
            "uploaded_at": datetime.datetime.utcnow().isoformat()
        })
        return blob_client.url
    except Exception as e:
        logging.error(f"Failed to upload JSON for {filename}: {e}")
        return ""


def download_pdfs(blob_service: BlobServiceClient, container: str, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    container_client = blob_service.get_container_client(container)
    paths: List[Path] = []
    for blob in container_client.list_blobs():
        if not blob.name.lower().endswith(".pdf"):
            continue
        local_path = output_dir / Path(blob.name).name
        try:
            with open(local_path, "wb") as f:
                f.write(container_client.get_blob_client(blob.name).download_blob().readall())
            paths.append(local_path)
            logging.info(f"Downloaded {blob.name}")
        except Exception as e:
            logging.warning(f"Failed to download {blob.name}: {e}")
    return paths


# ---------- Batch processing ----------
def batch_process(payer: str, pdf_container: str, json_container: str, temp_dir: str, config: Dict):
    blob_service = BlobServiceClient.from_connection_string(config["azure_conn"])
    ensure_container(blob_service, json_container)
    pdf_paths = download_pdfs(blob_service, pdf_container, Path(temp_dir))
    if not pdf_paths:
        logging.info("No PDFs found to process.")
        return

    stats = {"total": len(pdf_paths), "success": 0}
    max_workers = config["max_workers"]

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    extract_hybrid, pdf_path, payer, config, blob_service, json_container
                ): pdf_path
                for pdf_path in pdf_paths
            }
            for idx, future in enumerate(as_completed(futures), 1):
                ok = future.result()
                if ok:
                    stats["success"] += 1
                logging.info(f"[{idx}/{len(pdf_paths)}] {futures[future].name} -> {'ok' if ok else 'failed'}")
    else:
        for idx, pdf_path in enumerate(pdf_paths, 1):
            ok = extract_hybrid(pdf_path, payer, config, blob_service, json_container)
            if ok:
                stats["success"] += 1
            logging.info(f"[{idx}/{len(pdf_paths)}] {pdf_path.name} -> {'ok' if ok else 'failed'}")

    logging.info(f"Done. {stats['success']}/{stats['total']} PDFs processed successfully.")


# ---------- CLI ----------
def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Hybrid rule/AI policy extractor")
    parser.add_argument("payer", help="Payer name (used for uploads)")
    parser.add_argument("--pdf-container", default=os.getenv("AZURE_CONTAINER_NAME", "pdfs"))
    parser.add_argument("--json-container", default=os.getenv("AZURE_JSON_CONTAINER", "policy-json"))
    parser.add_argument("--temp-dir", default="./temp_pdfs_hybrid")
    args = parser.parse_args()

    load_dotenv()
    azure_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not azure_conn:
        raise SystemExit("AZURE_STORAGE_CONNECTION_STRING is required")

    hf_token = os.getenv("HF_TOKEN")
    hf_model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
    max_retries = int(os.getenv("MAX_RETRIES", "2"))
    ai_timeout = int(os.getenv("AI_TIMEOUT", "60"))
    enable_ai = os.getenv("ENABLE_AI_FALLBACK", "true").lower() != "false"
    max_workers = int(os.getenv("MAX_WORKERS", "1"))

    if enable_ai and not hf_token:
        raise SystemExit("HF_TOKEN is required when AI fallback is enabled")

    hf_client = InferenceClient(token=hf_token) if enable_ai else None

    config = {
        "azure_conn": azure_conn,
        "hf_client": hf_client,
        "hf_model": hf_model,
        "threshold": confidence_threshold,
        "max_retries": max_retries,
        "ai_timeout": ai_timeout,
        "enable_ai": enable_ai,
        "max_workers": max_workers,
    }

    batch_process(args.payer, args.pdf_container, args.json_container, args.temp_dir, config)


if __name__ == "__main__":
    main()
