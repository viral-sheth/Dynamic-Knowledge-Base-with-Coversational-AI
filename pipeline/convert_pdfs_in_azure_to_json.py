#!/usr/bin/env python3
"""
Convert PDFs already in Azure Blob Storage to JSON using a Hugging Face model,
then upload the JSON to a dedicated container.

Env vars:
- AZURE_STORAGE_CONNECTION_STRING (required)
- AZURE_CONTAINER_NAME (PDF container, default: pdfs)
- AZURE_JSON_CONTAINER (JSON container, default: policy-json)
- HUGGINGFACEHUB_API_TOKEN (required for HF Inference API)
- HF_POLICY_MODEL (optional, default: meta-llama/Meta-Llama-3-8B-Instruct)

Usage:
    python convert_pdfs_in_azure_to_json.py <payer_name>
"""

import os
import json
import argparse
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from huggingface_hub import InferenceClient
import fitz  # PyMuPDF
import PyPDF2


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text using PyMuPDF first, fallback to PyPDF2."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception:
        pass
    if text:
        return text
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception:
        pass
    return text


def chunk_text(text: str, chunk_size: int = 6000, overlap: int = 500) -> List[str]:
    """Chunk text to fit model context (approx by characters)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def validate_policy_json(data: Dict) -> bool:
    """Basic schema validation."""
    required = [
        "policy_id",
        "policy_type",
        "effective_date",
        "end_date",
        "supersedes",
        "summary",
        "payer_name",
        "source_pdf",
    ]
    if not isinstance(data, dict):
        return False
    for key in required:
        if key not in data:
            return False
    if not isinstance(data.get("supersedes"), list):
        data["supersedes"] = []
    return True


def generate_policy_json_with_hf(client: InferenceClient, text: str, payer_name: str, source_pdf: str) -> Dict:
    """Use HF model with chunking to extract policy JSON."""
    chunks = chunk_text(text)
    prompt_template = textwrap.dedent(
        """
        You are a healthcare payer policy extractor. Read the policy text and return ONLY compact JSON with keys:
        policy_id (string), policy_type (prior_auth|timely_filing|appeals|claims|billing|coverage|unknown),
        effective_date (YYYY-MM-DD or null), end_date (YYYY-MM-DD or null),
        supersedes (array of strings), summary (string), payer_name (string), source_pdf (string).

        Payer: "{payer}"
        Source PDF: "{pdf}"

        Policy text:
        {content}

        JSON only. No prose. No code fences.
        """
    ).strip()

    for idx, chunk in enumerate(chunks):
        prompt = prompt_template.format(payer=payer_name, pdf=source_pdf, content=chunk)
        try:
            resp = client.text_generation(
                prompt=prompt,
                max_new_tokens=700,
                temperature=0.15,
                do_sample=False,
                stop=["</s>", "```"],
            )
            cleaned = resp.strip().strip("`")
            candidate = json.loads(cleaned)
            if validate_policy_json(candidate):
                return candidate
        except Exception as e:
            print(f"HF chunk {idx} failed: {e}")
            continue
    return {}


def upload_json_to_azure(blob_service: BlobServiceClient, container: str, data: Dict, payer_name: str, source_pdf: str) -> str:
    """Upload JSON to Azure."""
    try:
        blob_name = f"{payer_name}/{Path(source_pdf).stem}.json"
        blob_client = blob_service.get_blob_client(container=container, blob=blob_name)
        blob_client.upload_blob(
            json.dumps(data, indent=2),
            overwrite=True,
            metadata={
                "payer_name": payer_name,
                "source_pdf": source_pdf,
                "uploaded_at": datetime.utcnow().isoformat(),
            },
        )
        return blob_client.url
    except Exception as e:
        print(f"Failed to upload JSON for {source_pdf}: {e}")
        return ""


def ensure_container(blob_service: BlobServiceClient, container: str):
    try:
        blob_service.create_container(container)
    except Exception:
        pass


def download_pdfs(blob_service: BlobServiceClient, container: str, output_dir: str) -> List[str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    container_client = blob_service.get_container_client(container)
    paths = []
    for blob in container_client.list_blobs():
        if not blob.name.lower().endswith(".pdf"):
            continue
        local_path = Path(output_dir) / Path(blob.name).name
        try:
            with open(local_path, "wb") as f:
                f.write(container_client.get_blob_client(blob.name).download_blob().readall())
            paths.append(str(local_path))
            print(f"Downloaded {blob.name}")
        except Exception as e:
            print(f"Failed to download {blob.name}: {e}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Convert Azure PDFs to JSON with HF model")
    parser.add_argument("payer", help="Payer name (used for JSON metadata/path)")
    parser.add_argument("--pdf-container", default=os.getenv("AZURE_CONTAINER_NAME", "pdfs"))
    parser.add_argument("--json-container", default=os.getenv("AZURE_JSON_CONTAINER", "policy-json"))
    parser.add_argument("--temp-dir", default="./temp_pdfs")
    args = parser.parse_args()

    load_dotenv()
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    hf_model = os.getenv("HF_POLICY_MODEL", "microsoft/Phi-3.5-mini-instruct")

    if not connection_string:
        raise SystemExit("AZURE_STORAGE_CONNECTION_STRING is required")
    if not hf_token:
        raise SystemExit("HUGGINGFACEHUB_API_TOKEN is required for HF inference")

    blob_service = BlobServiceClient.from_connection_string(connection_string)
    ensure_container(blob_service, args.json_container)
    client = InferenceClient(model=hf_model, token=hf_token)

    pdf_paths = download_pdfs(blob_service, args.pdf_container, args.temp_dir)
    print(f"Processing {len(pdf_paths)} PDFs for payer {args.payer}...")

    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f"No text extracted from {pdf_path}")
            continue
        policy_json = generate_policy_json_with_hf(client, text, args.payer, os.path.basename(pdf_path))
        if not policy_json:
            print(f"HF extraction failed for {pdf_path}")
            continue
        policy_json.setdefault("payer_name", args.payer)
        policy_json.setdefault("source_pdf", os.path.basename(pdf_path))
        url = upload_json_to_azure(blob_service, args.json_container, policy_json, args.payer, os.path.basename(pdf_path))
        if url:
            print(f"Uploaded JSON: {url}")


if __name__ == "__main__":
    main()
