#!/usr/bin/env python3
"""
Fixed production-ready policy extractor with anti-hallucination and validation.

Key fixes:
- Validates extracted deadlines against source text
- Enhanced prior authorization detection
- Separates claims from timely filing concerns
- Stronger anti-hallucination prompts
- Debug output for verification

Usage:
    python fixed_production_extractor.py /path/to/manual.pdf --payer "United Healthcare"
"""

import os
import json
import argparse
import datetime
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.backend_options import PdfBackendOptions


# ============================================================================
# PDF PARSING
# ============================================================================

def build_converter() -> DocumentConverter:
    """Build Docling converter."""
    format_options = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=ThreadedPdfPipelineOptions(
                do_ocr=False,
                force_backend_text=True,
                accelerator_options=AcceleratorOptions(device="cpu", num_threads=4),
            ),
            backend_options=PdfBackendOptions(
                enable_local_fetch=False,
                enable_remote_fetch=False
            ),
        )
    }
    return DocumentConverter(format_options=format_options)


def parse_pdf(pdf_path: Path) -> tuple:
    """Extract text from PDF."""
    converter = build_converter()
    doc = converter.convert(str(pdf_path))
    markdown = doc.document.export_to_markdown()
    pages = len(doc.document.pages)
    return markdown, pages


# ============================================================================
# TEXT CLEANING
# ============================================================================

def is_toc_line(line: str) -> bool:
    """Check if line is TOC."""
    line_lower = line.lower().strip()
    if "table of contents" in line_lower or "contents" == line_lower:
        return True
    if line.count('.') > 10 and re.search(r'\d+\s*$', line):
        return True
    if re.match(r'^.*\.{3,}\s*\d+\s*$', line):
        return True
    return False


def is_table_line(line: str) -> bool:
    """Check if line is table."""
    if line.count('|') >= 3:
        return True
    if re.match(r'^[\s\-|:]+$', line):
        return True
    return False


def is_page_number_line(line: str) -> bool:
    """Check if line is page number."""
    line_stripped = line.strip()
    if re.match(r'^page\s+\d+', line_stripped.lower()):
        return True
    if re.match(r'^\d+\s+(of|/)\s+\d+$', line_stripped):
        return True
    if re.match(r'^\d+$', line_stripped):
        return True
    return False


def aggressive_clean_text(text: str) -> str:
    """Remove TOC, tables, page numbers, headers/footers."""
    lines = text.split('\n')
    cleaned = []
    
    # Track repeated lines (headers/footers)
    line_counts = {}
    for line in lines:
        stripped = line.strip()
        if len(stripped) > 5:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1
    
    repeated_lines = {line for line, count in line_counts.items() if count >= 3}
    
    in_table = False
    table_skip_count = 0
    
    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            continue
        if is_toc_line(line):
            continue
        if is_page_number_line(line):
            continue
        if stripped in repeated_lines:
            continue
        if is_table_line(line):
            in_table = True
            table_skip_count = 0
            continue
        
        if in_table:
            table_skip_count += 1
            if table_skip_count > 5:
                in_table = False
            continue
        
        if len(stripped) < 30:
            continue
        
        cleaned.append(line)
    
    return '\n'.join(cleaned)


# ============================================================================
# SECTION DETECTION (ENHANCED FOR PA)
# ============================================================================

SECTION_KEYWORDS = {
    "claims": {
        "headers": [
            "claims", "claim submission", "claim filing", "billing",
            "reimbursement", "submitting claims"
        ],
        "content": [
            "submit a claim", "file a claim", "claim form",
            "billing procedure", "submit claims"
        ]
    },
    "appeals": {
        "headers": [
            "appeals", "grievances", "disputes", "complaints",
            "reconsideration", "appeal process"
        ],
        "content": [
            "file an appeal", "appeal process", "adverse determination",
            "dispute resolution", "grievance procedure"
        ]
    },
    "prior_authorization": {
        "headers": [
            "prior authorization", "preauthorization", "pre-authorization",
            "authorization", "authorization requirements", "precertification",
            "services requiring authorization", "obtaining authorization",
            "authorization process"
        ],
        "content": [
            "requires authorization", "must be authorized", "authorization required",
            "authorization form", "authorization request", "pre-cert",
            "obtain authorization", "authorization timeframe", "submit authorization",
            "authorization approval", "notify within", "notification requirement"
        ]
    },
    "timely_filing": {
        "headers": [
            "timely filing", "filing deadline", "filing timeframe",
            "submission deadline", "filing requirements", "filing limits"
        ],
        "content": [
            "filing deadline", "days from date of service",
            "submission timeframe", "time limit", "filing period"
        ]
    },
    "credentialing": {
        "headers": [
            "credentialing", "provider enrollment", "contracting",
            "provider application", "network participation"
        ],
        "content": [
            "provider application", "enrollment process",
            "credentialing application", "network participation"
        ]
    }
}


def find_section_content(text: str, section_name: str, context_lines: int = 200) -> Optional[str]:
    """Find section with enhanced detection (more lines for PA)."""
    keywords = SECTION_KEYWORDS.get(section_name, {})
    headers = keywords.get("headers", [])
    content_indicators = keywords.get("content", [])
    
    lines = text.split('\n')
    sections_found = []
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        is_header = any(kw in line_lower for kw in headers)
        is_content = any(kw in line_lower for kw in content_indicators)
        
        if is_header or is_content:
            start = i
            end = min(len(lines), i + context_lines)
            section_text = '\n'.join(lines[start:end])
            
            if len(section_text) > 500:
                sections_found.append(section_text)
    
    if not sections_found:
        return None
    
    combined = '\n\n'.join(sections_found)
    return combined[:15000]  # Increased for PA sections


def extract_all_sections(text: str) -> Dict[str, Optional[str]]:
    """Extract all sections with debug output."""
    sections = {}
    
    for section_name in SECTION_KEYWORDS.keys():
        print(f"    ├─ Searching for {section_name}...")
        content = find_section_content(text, section_name)
        
        if content:
            print(f"    │  ✓ Found {len(content)} chars")
            # Debug: show first 200 chars
            preview = content[:200].replace('\n', ' ')
            print(f"    │  Preview: {preview}...")
        else:
            print(f"    │  ✗ Not found")
        
        sections[section_name] = content
    
    return sections


# ============================================================================
# METADATA EXTRACTION
# ============================================================================

def classify_policy_type(text: str) -> str:
    """Classify document type."""
    text_lower = text.lower()
    
    manual_indicators = ["provider manual", "provider handbook", "provider guide"]
    if any(ind in text_lower for ind in manual_indicators):
        return "general"
    
    topic_counts = {
        "claims": text_lower.count("claims"),
        "appeals": text_lower.count("appeal"),
        "prior_authorization": text_lower.count("prior auth"),
        "timely_filing": text_lower.count("timely filing"),
        "credentialing": text_lower.count("credentialing"),
    }
    
    max_count = max(topic_counts.values())
    total_count = sum(topic_counts.values())
    
    if max_count > 0 and total_count > 0 and max_count / total_count > 0.6:
        return max(topic_counts, key=topic_counts.get)
    
    return "general"


def extract_dates(text: str) -> Dict[str, Optional[str]]:
    """Extract dates. Returns None if not found."""
    dates = {"effective_date": None, "expiration_date": None}
    
    patterns = {
        "effective": [
            r"effective\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
            r"effective[:\s]+(\w+\s+\d{1,2},?\s+\d{4})",
        ],
        "expiration": [
            r"expir(?:ation|es)\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
            r"valid\s+through[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
        ]
    }
    
    text_lower = text.lower()
    
    for date_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, text_lower)
            if match:
                dates[f"{date_type}_date"] = match.group(1)
                break
    
    return dates


# ============================================================================
# DEADLINE VALIDATION (ANTI-HALLUCINATION)
# ============================================================================

def validate_deadlines_in_source(deadlines: List[str], source_text: str) -> List[str]:
    """
    Validate extracted deadlines against source text.
    Remove any deadlines whose numbers don't appear in source.
    """
    validated = []
    source_lower = source_text.lower()
    
    for deadline in deadlines:
        # Extract all numbers from deadline
        numbers = re.findall(r'\b(\d+)\b', deadline)
        
        # Check if those numbers + "day" appear in source
        found_in_source = False
        for num in numbers:
            patterns = [
                f"{num} day",
                f"{num} calendar day",
                f"{num} business day",
                f"{num}-day",
            ]
            if any(pattern in source_lower for pattern in patterns):
                found_in_source = True
                break
        
        if found_in_source or not numbers:  # Keep if validated or no numbers
            validated.append(deadline)
        else:
            print(f"      ⚠️  Removed hallucinated deadline: '{deadline}'")
    
    return validated


# ============================================================================
# AI EXTRACTION WITH SECTION-SPECIFIC PROMPTS
# ============================================================================

def build_claims_prompt(section_text: str, payer: str) -> str:
    """Claims-specific prompt (NO filing deadlines)."""
    return f"""Extract claims submission requirements from this {payer} provider manual section.

FOCUS ONLY ON:
- Submission methods (electronic, paper, portal, clearinghouse)
- Required information on forms (NPI, tax ID, codes)
- Documentation requirements (medical records, PA numbers)
- Claim forms to use (CMS-1500, UB-04)
- Coding requirements (CPT, HCPCS, ICD-10, modifiers)
- How to submit corrected/resubmitted claims

DO NOT EXTRACT:
- Filing deadlines (those go in timely_filing section)
- Prior authorization requirements (those go in prior_authorization section)

Return ONLY valid JSON:

{{
  "requirements": [
    "Specific submission requirement with details"
  ],
  "forms": [
    "Form name or portal"
  ],
  "notes": [
    "Important clarification"
  ]
}}

CRITICAL: Extract ONLY what is explicitly stated. Do not infer or guess.

Text:
{section_text[:10000]}

Return JSON only."""


def build_timely_filing_prompt(section_text: str, payer: str) -> str:
    """Timely filing-specific prompt."""
    return f"""Extract timely filing deadlines from this {payer} provider manual section.

FOCUS ONLY ON:
- Filing deadlines by provider type (participating vs non-participating)
- Filing deadlines by claim type (initial, corrected, resubmitted)
- Special timeframes (emergency, retroactive)
- What triggers the deadline (date of service, denial date)
- Payment processing timeframes

Return ONLY valid JSON:

{{
  "deadlines": [
    "X days from date of service for [provider type]",
    "Y days for [claim type]"
  ],
  "requirements": [
    "Calculate deadline from date of service",
    "Rules about when deadline starts"
  ],
  "notes": [
    "Exceptions or special circumstances"
  ]
}}

CRITICAL ANTI-HALLUCINATION RULES:
1. Extract ONLY deadlines explicitly stated in the text
2. If you see "90 days", write "90 days" (not 180 or any other number)
3. If you see "365 days" or "1 year", use the exact wording
4. Include provider type distinctions (participating vs non-participating)
5. Do NOT make up deadlines - if unclear, omit it

Text:
{section_text[:10000]}

Return JSON only."""


def build_prior_auth_prompt(section_text: str, payer: str) -> str:
    """Prior authorization-specific prompt."""
    return f"""Extract prior authorization requirements from this {payer} provider manual section.

FOCUS ON:
- Which services/procedures require prior authorization
- How to submit authorization requests (portal, phone, fax)
- Timeframes for authorization (notification windows, decision timeframes)
- Forms or reference numbers
- Pre-service vs post-service notification requirements
- Contact information for authorization

Return ONLY valid JSON:

{{
  "requirements": [
    "Specific service that requires PA",
    "Another service requiring PA"
  ],
  "procedures": [
    "How to submit PA request",
    "Where to submit request"
  ],
  "deadlines": [
    "X hours/days for notification",
    "Y days for authorization decision"
  ],
  "forms": [
    "Form name or portal"
  ],
  "notes": [
    "Important exceptions"
  ]
}}

CRITICAL: Extract ONLY what is stated. Do not assume or infer.

Text:
{section_text[:10000]}

Return JSON only."""


def build_generic_prompt(section_name: str, section_text: str, payer: str) -> str:
    """Generic prompt for appeals and credentialing."""
    return f"""Extract {section_name} requirements from this {payer} provider manual section.

Return ONLY valid JSON:

{{
  "requirements": [
    "Specific requirement with details"
  ],
  "deadlines": [
    "X days for Y"
  ],
  "forms": [
    "Form name"
  ],
  "notes": [
    "Important clarification"
  ]
}}

CRITICAL: Extract ONLY information explicitly stated. Do not guess.

Text:
{section_text[:10000]}

Return JSON only."""


def get_section_prompt(section_name: str, section_text: str, payer: str) -> str:
    """Get appropriate prompt for section."""
    if section_name == "claims":
        return build_claims_prompt(section_text, payer)
    elif section_name == "timely_filing":
        return build_timely_filing_prompt(section_text, payer)
    elif section_name == "prior_authorization":
        return build_prior_auth_prompt(section_text, payer)
    else:
        return build_generic_prompt(section_name, section_text, payer)


def clean_json_response(content: str) -> str:
    """Clean AI response."""
    content = content.replace("```json", "").replace("```", "").strip()
    first_brace = content.find('{')
    if first_brace > 0:
        content = content[first_brace:]
    last_brace = content.rfind('}')
    if last_brace > 0:
        content = content[:last_brace + 1]
    return content


def validate_extraction(result: Dict, section_name: str) -> bool:
    """Validate extraction has content."""
    if section_name == "claims":
        required = ["requirements", "forms", "notes"]
    elif section_name == "timely_filing":
        required = ["deadlines", "requirements", "notes"]
    elif section_name == "prior_authorization":
        required = ["requirements", "procedures", "deadlines", "forms", "notes"]
    else:
        required = ["requirements", "deadlines", "forms", "notes"]
    
    if not all(field in result for field in required):
        return False
    if not all(isinstance(result[field], list) for field in required):
        return False
    
    # For PA, must have at least requirements or procedures
    if section_name == "prior_authorization":
        return len(result["requirements"]) > 0 or len(result["procedures"]) > 0
    
    total_items = sum(len(result.get(field, [])) for field in required)
    return total_items >= 2


def extract_section_validated(
    section_name: str,
    section_text: Optional[str],
    payer: str,
    client: OpenAI,
    model: str,
    max_retries: int = 3
) -> Dict:
    """Extract with validation and anti-hallucination checks."""
    
    # Default empty structure
    if section_name == "claims":
        empty = {"requirements": [], "forms": [], "notes": []}
    elif section_name == "timely_filing":
        empty = {"deadlines": [], "requirements": [], "notes": []}
    elif section_name == "prior_authorization":
        empty = {"requirements": [], "procedures": [], "deadlines": [], "forms": [], "notes": []}
    else:
        empty = {"requirements": [], "deadlines": [], "forms": [], "notes": []}
    
    if not section_text or len(section_text) < 200:
        return empty
    
    for attempt in range(max_retries):
        try:
            prompt = get_section_prompt(section_name, section_text, payer)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Extract policy requirements as JSON only. Be factual, do not hallucinate."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=4000,
            )
            
            content = response.choices[0].message.content.strip()
            content = clean_json_response(content)
            result = json.loads(content)
            
            # Validate deadlines against source
            if "deadlines" in result:
                result["deadlines"] = validate_deadlines_in_source(
                    result["deadlines"], 
                    section_text
                )
            
            if validate_extraction(result, section_name):
                return result
            else:
                print(f"      ⚠️  Validation failed, retry {attempt + 1}")
            
        except json.JSONDecodeError as e:
            print(f"      ⚠️  JSON error: {e}, retry {attempt + 1}")
        except Exception as e:
            print(f"      ⚠️  Error: {e}, retry {attempt + 1}")
        
        if attempt < max_retries - 1:
            time.sleep(2)
    
    print(f"      ✗ All retries failed for {section_name}")
    return empty


# ============================================================================
# MAIN EXTRACTION
# ============================================================================

def extract_policy_fixed(
    pdf_path: Path,
    payer: str,
    client: OpenAI,
    model: str
) -> Dict:
    """Fixed extraction with validation."""
    
    print(f"\n{'='*70}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*70}\n")
    
    print("Step 1: Parsing PDF...")
    raw_text, pages = parse_pdf(pdf_path)
    print(f"  ✓ {len(raw_text)} chars from {pages} pages\n")
    
    print("Step 2: Cleaning text...")
    cleaned_text = aggressive_clean_text(raw_text)
    print(f"  ✓ Cleaned to {len(cleaned_text)} chars\n")
    
    print("Step 3: Classifying...")
    policy_type = classify_policy_type(cleaned_text)
    print(f"  ✓ Type: {policy_type}\n")
    
    print("Step 4: Extracting dates...")
    dates = extract_dates(cleaned_text)
    print(f"  ✓ Effective: {dates['effective_date']}")
    print(f"  ✓ Expiration: {dates['expiration_date']}\n")
    
    print("Step 5: Detecting sections...")
    sections = extract_all_sections(cleaned_text)
    sections_found = [k for k, v in sections.items() if v is not None]
    print(f"\n  ✓ Found {len(sections_found)}/{len(SECTION_KEYWORDS)} sections\n")
    
    print("Step 6: Extracting with validation...\n")
    section_details = {}
    
    for section_name in SECTION_KEYWORDS.keys():
        section_text = sections.get(section_name)
        
        if section_text:
            print(f"  Extracting {section_name}...")
            details = extract_section_validated(
                section_name, section_text, payer, client, model
            )
            
            # Count items
            total = sum(len(v) for v in details.values() if isinstance(v, list))
            print(f"    ✓ Extracted {total} total items\n")
            
            section_details[section_name] = details
        else:
            print(f"  Skipping {section_name} (not found)\n")
            if section_name == "claims":
                section_details[section_name] = {"requirements": [], "forms": [], "notes": []}
            elif section_name == "timely_filing":
                section_details[section_name] = {"deadlines": [], "requirements": [], "notes": []}
            elif section_name == "prior_authorization":
                section_details[section_name] = {"requirements": [], "procedures": [], "deadlines": [], "forms": [], "notes": []}
            else:
                section_details[section_name] = {"requirements": [], "deadlines": [], "forms": [], "notes": []}
    
    print("Step 7: Building output...\n")
    
    summary = f"{payer} provider manual" if policy_type == "general" else f"{payer} {policy_type} policy"
    if sections_found:
        summary += f" covering {', '.join(sections_found)}"
    
    result = {
        "filename": pdf_path.name,
        "payer": payer,
        "policy_type": policy_type,
        "title": f"{payer} Provider Manual" if policy_type == "general" else f"{payer} {policy_type.replace('_', ' ').title()} Policy",
        "effective_date": dates["effective_date"],
        "expiration_date": dates["expiration_date"],
        "summary": summary,
        "sections": section_details,
        "metadata": {
            "pages": pages,
            "sections_found": sections_found,
            "extraction_method": "validated+anti-hallucination",
            "model_used": model,
            "extraction_date": datetime.date.today().isoformat(),
        }
    }
    
    return result


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fixed policy extractor")
    parser.add_argument("pdf_path", help="Path to PDF")
    parser.add_argument("--payer", required=True, help="Payer name")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    args = parser.parse_args()
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("❌ OPENAI_API_KEY required")
    
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        raise SystemExit(f"❌ PDF not found: {pdf_path}")
    
    client = OpenAI(api_key=api_key)
    result = extract_policy_fixed(pdf_path, args.payer, client, args.model)
    
    print("="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"\n📋 Type: {result['policy_type']}")
    print(f"📄 Pages: {result['metadata']['pages']}")
    print(f"✓ Sections: {', '.join(result['metadata']['sections_found'])}")
    
    print("\n" + "="*70)
    print("RESULTS BY SECTION")
    print("="*70)
    
    for section_name, details in result['sections'].items():
        items = sum(len(v) for v in details.values() if isinstance(v, list))
        if items > 0:
            print(f"\n{section_name.replace('_', ' ').title()}: {items} items")
            for key, value in details.items():
                if isinstance(value, list) and len(value) > 0:
                    print(f"  {key}: {len(value)}")
                    for item in value[:3]:
                        print(f"    - {item}")
                    if len(value) > 3:
                        print(f"    ... and {len(value) - 3} more")
    
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✅ Saved to: {out_path}")
    else:
        print("\n" + "="*70)
        print("JSON OUTPUT")
        print("="*70)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()