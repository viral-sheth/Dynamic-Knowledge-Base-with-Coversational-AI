#!/usr/bin/env python3
"""
Convert PDFs stored in Azure to structured policy JSON using Docling (layout-aware parsing)
and Groq (Llama 3.1) for policy-field extraction. Uploads JSON back to Azure.

Env vars required:
- AZURE_STORAGE_CONNECTION_STRING
- GROQ_API_KEY

Optional env vars:
- AZURE_CONTAINER_NAME (PDF container, default: pdfs)
- AZURE_JSON_CONTAINER (JSON container, default: policy-json)
- GROQ_MODEL (default: llama-3.1-70b-versatile; or use llama-3.1-8b-instant for speed)
"""

import os
import json
import argparse
import textwrap
import datetime
from pathlib import Path
from typing import Dict, List
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from groq import Groq
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.backend_options import PdfBackendOptions


def ensure_container(blob_service: BlobServiceClient, container: str):
    try:
        blob_service.create_container(container)
    except Exception:
        pass


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
            print(f"Downloaded {blob.name}")
        except Exception as e:
            print(f"Failed to download {blob.name}: {e}")
    return paths


def docling_parse(converter: DocumentConverter, pdf_path: Path) -> Dict:
    doc = converter.convert(str(pdf_path))
    markdown = doc.document.export_to_markdown()
    tables_md = []
    for table in doc.document.tables:
        try:
            tables_md.append(table.export_to_markdown())
        except Exception:
            continue
    pages = len(doc.document.pages)
    return {"markdown": markdown, "tables": tables_md, "pages": pages}


def build_prompt(markdown: str, tables: List[str], payer: str, filename: str) -> str:
    tables_joined = "\n\n".join(tables) if tables else "No tables."
    return textwrap.dedent(f"""
    You are a healthcare payer policy extractor. Read the provided policy text and tables and return ONLY valid JSON (no prose, no markdown, no code fences) matching this schema:
    {{
      "filename": "...",
      "payer": "...",
      "policy_type": "claims|appeals|prior_authorization|timely_filing|credentialing|general",
      "title": "...",
      "effective_date": "YYYY-MM-DD or null",
      "expiration_date": "YYYY-MM-DD or null",
      "summary": "Brief 1-2 sentence summary",
      "key_requirements": ["..."],
      "timely_filing_days": integer or null,
      "prior_auth_required": true|false|null,
      "content": "full extracted text used for decision",
      "tables": ["table1 markdown", "table2 markdown"],
      "metadata": {{
        "pages": integer,
        "extraction_method": "docling+groq",
        "extracted_date": "YYYY-MM-DD"
      }}
    }}

    If a field is missing, use null (or empty list for key_requirements). Choose policy_type from the enum above. Respond with JSON only.
    Payer: {payer}
    Filename: {filename}

    Policy text (markdown):
    {markdown}

    Tables (markdown):
    {tables_joined}
    """).strip()


def groq_extract(client: Groq, prompt: str, model: str) -> Dict:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1200,
        )
        text = resp.choices[0].message.content
        return json.loads(text)
    except Exception as e:
        print(f"Groq extraction failed ({model}): {e}")
        return {}


def ollama_extract(prompt: str, model: str, host: str, timeout: int = 60, retries: int = 2) -> Dict:
    """Call local Ollama chat API and return parsed JSON."""
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1200
        }
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            message = data.get("message", {}) or data.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")
            return json.loads(content)
        except Exception as e:
            if attempt == retries:
                print(f"Ollama extraction failed after {attempt} attempts: {e}")
                return {}
            else:
                print(f"Ollama attempt {attempt} failed: {e}, retrying...")


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
        print(f"Failed to upload JSON for {filename}: {e}")
        return ""


def build_converter():
    format_options = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=ThreadedPdfPipelineOptions(
                do_ocr=False,
                force_backend_text=True,
                accelerator_options=AcceleratorOptions(device="cpu", num_threads=4)
            ),
            backend_options=PdfBackendOptions(enable_local_fetch=False, enable_remote_fetch=False),
        )
    }
    return DocumentConverter(format_options=format_options)


def process_pdf(pdf_path: Path, payer: str, blob_service: BlobServiceClient, json_container: str,
                groq_client: Groq, groq_model: str, use_ollama: bool, ollama_model: str, ollama_host: str):
    try:
        converter = build_converter()
        parsed = docling_parse(converter, pdf_path)
        prompt = build_prompt(parsed["markdown"], parsed["tables"], payer, pdf_path.name)

        if use_ollama:
            policy_json = ollama_extract(prompt, ollama_model, ollama_host)
        else:
            policy_json = groq_extract(groq_client, prompt, groq_model)

        if not policy_json:
            print(f"Extraction failed for {pdf_path.name}")
            return False

        policy_json.setdefault("filename", pdf_path.name)
        policy_json.setdefault("payer", payer)
        policy_json.setdefault("metadata", {})
        policy_json["metadata"].setdefault("pages", parsed["pages"])
        policy_json["metadata"]["extraction_method"] = "docling+ollama" if use_ollama else "docling+groq"
        policy_json["metadata"]["extracted_date"] = datetime.date.today().isoformat()
        policy_json.setdefault("tables", parsed["tables"])
        policy_json.setdefault("content", parsed["markdown"])

        url = upload_json(blob_service, json_container, payer, pdf_path.name, policy_json)
        if url:
            print(f"Uploaded JSON: {url}")
            return True
        return False
    except Exception as e:
        print(f"Failed processing {pdf_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert Azure PDFs to policy JSON using Docling + Groq.")
    parser.add_argument("payer", help="Payer name (used in output JSON)")
    parser.add_argument("--pdf-container", default=os.getenv("AZURE_CONTAINER_NAME", "pdfs"))
    parser.add_argument("--json-container", default=os.getenv("AZURE_JSON_CONTAINER", "policy-json"))
    parser.add_argument("--temp-dir", default="./temp_pdfs_docling")
    args = parser.parse_args()

    load_dotenv()
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    groq_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    max_workers = int(os.getenv("MAX_WORKERS", "1"))

    if not connection_string:
        raise SystemExit("AZURE_STORAGE_CONNECTION_STRING is required")
    if not use_ollama and not groq_key:
        raise SystemExit("GROQ_API_KEY is required when USE_OLLAMA is false")

    blob_service = BlobServiceClient.from_connection_string(connection_string)
    ensure_container(blob_service, args.json_container)

    groq_client = Groq(api_key=groq_key) if not use_ollama else None

    pdf_paths = download_pdfs(blob_service, args.pdf_container, Path(args.temp_dir))
    if not pdf_paths:
        print("No PDFs found to process.")
        return

    success = 0
    total = len(pdf_paths)
    print(f"Processing {total} PDFs for payer {args.payer} (Ollama={use_ollama}) ...")

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_pdf,
                    pdf_path,
                    args.payer,
                    blob_service,
                    args.json_container,
                    groq_client,
                    groq_model,
                    use_ollama,
                    ollama_model,
                    ollama_host
                ): pdf_path
                for pdf_path in pdf_paths
            }
            for idx, future in enumerate(as_completed(futures), 1):
                pdf_path = futures[future]
                ok = future.result()
                if ok:
                    success += 1
                print(f"[{idx}/{total}] {pdf_path.name} -> {'ok' if ok else 'failed'}")
    else:
        for idx, pdf_path in enumerate(pdf_paths, 1):
            ok = process_pdf(
                pdf_path,
                args.payer,
                blob_service,
                args.json_container,
                groq_client,
                groq_model,
                use_ollama,
                ollama_model,
                ollama_host
            )
            if ok:
                success += 1
            print(f"[{idx}/{total}] {pdf_path.name} -> {'ok' if ok else 'failed'}")

    print(f"Done. {success}/{total} PDFs processed successfully.")


if __name__ == "__main__":
    main()
