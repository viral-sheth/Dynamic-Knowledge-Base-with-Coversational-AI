#!/usr/bin/env python3
"""
Targeted Healthcare Rule Crawler
Specifically targets PDFs containing: Timely Filing, Prior Authorization, Billing Nuances, and Appeals
"""

import time
import logging
import json
import os
import hashlib
import textwrap
from datetime import datetime
from collections import deque
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import Dict, List

import requests
import fitz  # PyMuPDF
import PyPDF2
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from azure.storage.blob import BlobServiceClient

from pipeline.azure_pdf_uploader import AzurePDFUploader
from pipeline.policy_deduplication_system import PolicyDeduplicationEngine

class TargetedHealthcareRuleCrawler:
    """Focused crawler for specific healthcare rule types"""
    
    def __init__(self, headless=True, max_depth=3):
        self.headless = headless
        self.max_depth = max_depth
        self.visited_urls = set()
        self.discovered_pdfs = []
        # Extra terms to follow even if not matching specific rule keywords
        self.extra_link_terms = [
            'policy', 'policies', 'bulletin', 'bulletins', 'manual', 'guide',
            'guidelines', 'criteria', 'protocol', 'coverage', 'medical policy',
            'clinical policy', 'provider manual'
        ]
        # Hard filters to avoid off-topic clinical policies
        self.excluded_terms = [
            'genetic', 'molecular', 'lab test', 'laboratory', 'drug list',
            'formulary', 'immunization', 'vaccine', 'dental', 'oral health',
            'hearing aid', 'vision', 'prosthetic', 'sleep apnea', 'cpap',
            'clinical policy bulletin', 'medical policy'
        ]
        
        # Target rule types and their keywords
        self.target_rules = {
            'timely_filing': {
                'keywords': [
                    'timely filing', 'filing deadline', 'filing requirement', 'filing limit',
                    'claim submission deadline', 'submission timeframe', 'filing window',
                    'claim filing', 'timely submission', 'deadline requirement'
                ],
                'url_patterns': [
                    'timely', 'filing', 'deadline', 'submission'
                ]
            },
            'prior_authorization': {
                'keywords': [
                    'prior authorization', 'preauthorization', 'pre-authorization', 
                    'prior auth', 'preauth', 'medical necessity', 'coverage determination',
                    'pre-approval', 'precertification', 'authorization required',
                    'prior approval', 'authorization request', 'coverage approval'
                ],
                'url_patterns': [
                    'prior', 'auth', 'preauth', 'approval', 'precert', 'authorization'
                ]
            },
            'billing_nuances': {
                'keywords': [
                    'billing', 'reimbursement', 'payment', 'claim processing',
                    'billing guidelines', 'coding', 'billing procedures', 'fee schedule',
                    'billing requirements', 'claim submission', 'billing policies',
                    'payment policies', 'reimbursement rates', 'billing rules',
                    'provider manual', 'provider guide'
                ],
                'url_patterns': [
                    'billing', 'payment', 'reimbursement', 'claims', 'coding', 'fees'
                ]
            },
            'appeals': {
                'keywords': [
                    'appeal', 'appeals', 'grievance', 'dispute resolution', 
                    'claim dispute', 'complaint', 'reconsideration', 'external review',
                    'appeal process', 'grievance procedure', 'dispute process',
                    'appeal rights', 'complaint resolution', 'review process'
                ],
                'url_patterns': [
                    'appeal', 'grievance', 'dispute', 'complaint', 'review'
                ]
            }
        }
        
        self.processing_log = {
            "crawl_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_rules": list(self.target_rules.keys()),
            "companies_processed": [],
            "total_targeted_pdfs": 0,
            "pdfs_by_rule_type": {}
        }
        
        self.setup_webdriver()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Azure + HF setup
        load_dotenv()
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        pdf_container = os.getenv("AZURE_CONTAINER_NAME", "pdfs")
        self.json_container = os.getenv("AZURE_JSON_CONTAINER", "policy-json")
        self.azure_uploader = None
        self._existing_hashes_cache = {}
        self.blob_service = None
        self.dedup_engine = None

        if connection_string:
            try:
                self.azure_uploader = AzurePDFUploader(connection_string, pdf_container)
                self.blob_service = BlobServiceClient.from_connection_string(connection_string)
                self._ensure_container(self.json_container)
                self.dedup_engine = PolicyDeduplicationEngine(connection_string)
                self.logger.info(f"Azure uploader configured for container '{pdf_container}', JSON container '{self.json_container}'")
            except Exception as e:
                self.logger.warning(f"Azure init failed: {e}")

        # HF client
        hf_model = os.getenv("HF_POLICY_MODEL", "microsoft/Phi-3.5-mini-instruct")
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.hf_client = None
        if hf_token:
            try:
                self.hf_client = InferenceClient(model=hf_model, token=hf_token)
                self.logger.info(f"Hugging Face client initialized: {hf_model}")
            except Exception as e:
                self.logger.warning(f"HF client init failed: {e}")
    
    def setup_webdriver(self):
        """Setup Chrome WebDriver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.set_page_load_timeout(30)
    
    def close(self):
        if self.driver:
            self.driver.quit()

    def _ensure_container(self, container_name: str):
        if not self.blob_service:
            return
        try:
            self.blob_service.create_container(container_name)
        except Exception:
            pass
    
    def load_payer_configurations(self):
        """Load healthcare payer configurations with targeted URLs"""
        return {
            "united_healthcare": {
                "name": "United Healthcare",
                "base_url": "https://www.uhcprovider.com/",
                "targeted_urls": [
                    "https://www.uhcprovider.com/en/prior-authorization.html",
                    "https://www.uhcprovider.com/en/claims-payments.html",
                    "https://www.uhcprovider.com/en/policies-protocols.html",
                    "https://www.uhcprovider.com/en/admin-guides.html",
                    "https://www.uhcprovider.com/en/resource-library/appeals-grievances.html"
                ],
                "allowed_domains": ["uhcprovider.com", "unitedhealthcareonline.com"]
            },
            "anthem": {
                "name": "Anthem / Elevance Health",
                "base_url": "https://providers.anthem.com/",
                "targeted_urls": [
                    "https://providers.anthem.com/prior-authorization/",
                    "https://providers.anthem.com/claims-payment/",
                    "https://providers.anthem.com/appeals-grievances/",
                    "https://providers.anthem.com/provider-support/billing/",
                    "https://providers.anthem.com/docs/"
                ],
                "allowed_domains": ["anthem.com", "providers.anthem.com"]
            },
            "aetna": {
                "name": "Aetna",
                "base_url": "https://www.aetna.com/health-care-professionals/",
                "targeted_urls": [
                    "https://www.aetna.com/health-care-professionals/prior-authorization.html",
                    "https://www.aetna.com/health-care-professionals/claims-payment.html",
                    "https://www.aetna.com/health-care-professionals/appeals-grievances.html",
                    "https://www.aetna.com/health-care-professionals/billing-payment.html",
                    "https://www.aetna.com/health-care-professionals/clinical-policy-bulletins.html",
                    "https://www.aetna.com/health-care-professionals/medical-policies.html",
                    "https://www.aetna.com/health-care-professionals/document-library.html"
                ],
                "allowed_domains": ["aetna.com", "aetnabetterhealth.com", "aetna-better-health.com", "cvscaremark.com"]
            },
            "humana": {
                "name": "Humana",
                "base_url": "https://www.humana.com/provider/",
                "targeted_urls": [
                    "https://www.humana.com/provider/medical-resources/prior-authorization",
                    "https://www.humana.com/provider/claims",
                    "https://www.humana.com/provider/appeals",
                    "https://www.humana.com/provider/billing"
                ],
                "allowed_domains": ["humana.com"]
            },
            "cigna": {
                "name": "Cigna",
                "base_url": "https://www.cigna.com/healthcare-providers/",
                "targeted_urls": [
                    "https://www.cigna.com/healthcare-providers/prior-authorization",
                    "https://www.cigna.com/healthcare-providers/coverage-and-claims",
                    "https://www.cigna.com/healthcare-providers/appeals-grievances",
                    "https://www.cigna.com/healthcare-providers/billing-payment"
                ],
                "allowed_domains": ["cigna.com"]
            },
            "countycare": {
                "name": "CountyCare Health Plan",
                "base_url": "https://www.countycare.com/providers/",
                "targeted_urls": [
                    "https://www.countycare.com/providers/prior-authorization/",
                    "https://www.countycare.com/providers/claims-billing/",
                    "https://www.countycare.com/providers/appeals-grievances/",
                    "https://www.countycare.com/providers/manuals-and-resources/"
                ],
                "allowed_domains": ["countycare.com"]
            }
        }
    
    def is_pdf_url(self, url):
        """Check if URL points to a PDF"""
        return url.lower().endswith('.pdf') or '.pdf' in url.lower()
    
    def classify_pdf_relevance(self, text, href, url):
        """Classify PDF relevance to target rule types"""
        combined_text = f"{text} {href} {url}".lower()

        # Drop immediately if it contains excluded terms
        if any(term in combined_text for term in self.excluded_terms):
            return {
                'relevant_rules': [],
                'primary_rule': None,
                'scores': {},
                'is_targeted': False
            }
        
        relevant_rules = []
        scores = {}
        baseline_terms = ['provider manual', 'manual', 'provider guide', 'handbook']
        
        for rule_type, rule_data in self.target_rules.items():
            score = 0
            
            # Check keywords
            for keyword in rule_data['keywords']:
                if keyword in combined_text:
                    score += 2
            
            # Check URL patterns
            for pattern in rule_data['url_patterns']:
                if pattern in combined_text:
                    score += 1
            
            if score > 0:
                relevant_rules.append(rule_type)
                scores[rule_type] = score
        
        # If nothing matched, consider baseline provider manuals only
        if not scores and any(term in combined_text for term in baseline_terms):
            scores['billing_nuances'] = 2  # bump to meet min_score
            relevant_rules.append('billing_nuances')
        
        # Determine primary rule type (highest score)
        primary_rule = max(scores.keys(), key=lambda x: scores[x]) if scores else None

        # Require a minimum score of 2 to keep (avoid weak matches)
        min_score = 2
        if primary_rule and scores[primary_rule] < min_score:
            return {
                'relevant_rules': [],
                'primary_rule': None,
                'scores': {},
                'is_targeted': False
            }

        return {
            'relevant_rules': relevant_rules,
            'primary_rule': primary_rule,
            'scores': scores,
            'is_targeted': len(relevant_rules) > 0
        }

    def _load_existing_hashes(self, payer_name: str):
        """Load existing content hashes for a payer from Azure to avoid re-uploads"""
        if not self.azure_uploader:
            return set()
        if payer_name in self._existing_hashes_cache:
            return self._existing_hashes_cache[payer_name]
        hashes = set()
        try:
            container = self.azure_uploader.blob_service_client.get_container_client(
                self.azure_uploader.container_name
            )
            prefix = f"{payer_name}/"
            for blob in container.list_blobs(name_starts_with=prefix):
                meta = getattr(blob, "metadata", None) or {}
                content_hash = meta.get("content_hash")
                if content_hash:
                    hashes.add(content_hash)
        except Exception as e:
            self.logger.warning(f"Could not load existing hashes for {payer_name}: {e}")
        self._existing_hashes_cache[payer_name] = hashes
        return hashes

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF first, fallback to PyPDF2"""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            self.logger.debug(f"PyMuPDF failed on {pdf_path}: {e}")
        if text:
            return text
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            self.logger.debug(f"PyPDF2 failed on {pdf_path}: {e}")
        return text

    def _chunk_text(self, text: str, chunk_size: int = 6000, overlap: int = 500) -> List[str]:
        """Chunk text to fit model context (approx by characters)"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    def _validate_policy_json(self, data: Dict) -> bool:
        required = ["policy_id", "policy_type", "effective_date", "end_date", "supersedes", "summary", "payer_name", "source_pdf"]
        if not isinstance(data, dict):
            return False
        for key in required:
            if key not in data:
                return False
        if not isinstance(data.get("supersedes"), list):
            data["supersedes"] = []
        return True

    def generate_policy_json_with_hf(self, text: str, payer_name: str, source_pdf: str) -> Dict:
        """Use HF model with chunking to extract policy JSON"""
        if not self.hf_client:
            return {}

        chunks = self._chunk_text(text)
        prompt_template = textwrap.dedent("""
        You are a healthcare payer policy extractor. Read the policy text and return ONLY compact JSON with keys:
        policy_id (string), policy_type (prior_auth|timely_filing|appeals|claims|billing|coverage|unknown),
        effective_date (YYYY-MM-DD or null), end_date (YYYY-MM-DD or null),
        supersedes (array of strings), summary (string), payer_name (string), source_pdf (string).

        Payer: "{payer}"
        Source PDF: "{pdf}"

        Policy text:
        {content}

        JSON only. No prose. No code fences.
        """).strip()

        for idx, chunk in enumerate(chunks):
            prompt = prompt_template.format(payer=payer_name, pdf=source_pdf, content=chunk)
            try:
                resp = self.hf_client.text_generation(
                    prompt=prompt,
                    max_new_tokens=700,
                    temperature=0.15,
                    do_sample=False,
                    stop=["</s>", "```"]
                )
                cleaned = resp.strip().strip("`")
                candidate = json.loads(cleaned)
                if self._validate_policy_json(candidate):
                    return candidate
            except Exception as e:
                self.logger.debug(f"HF chunk {idx} failed: {e}")
                continue
        return {}

    def upload_json_to_azure(self, data: Dict, payer_name: str, source_pdf: str) -> str:
        """Upload extracted JSON to dedicated container"""
        if not self.blob_service:
            return ""
        try:
            blob_name = f"{payer_name}/{os.path.splitext(os.path.basename(source_pdf))[0]}.json"
            blob_client = self.blob_service.get_blob_client(
                container=self.json_container,
                blob=blob_name
            )
            blob_client.upload_blob(
                json.dumps(data, indent=2),
                overwrite=True,
                metadata={
                    "payer_name": payer_name,
                    "source_pdf": source_pdf,
                    "uploaded_at": datetime.utcnow().isoformat()
                }
            )
            return blob_client.url
        except Exception as e:
            self.logger.warning(f"Failed to upload JSON for {source_pdf}: {e}")
            return ""

    def convert_pdf_to_policy_json(self, pdf_path: str, payer_name: str, source_url: str, primary_rule: str) -> Dict:
        """Extract text, call HF for JSON, validate, upload JSON, dedup/save metadata"""
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            self.logger.warning(f"No text extracted from {pdf_path}")
            return {}

        policy_json = self.generate_policy_json_with_hf(text, payer_name, os.path.basename(pdf_path))

        if not policy_json:
            self.logger.warning(f"HF extraction failed for {pdf_path}")
            return {}

        # Ground mandatory fields
        policy_json.setdefault("payer_name", payer_name)
        policy_json.setdefault("source_pdf", os.path.basename(pdf_path))
        policy_json.setdefault("policy_type", primary_rule or policy_json.get("policy_type", "unknown"))

        # Upload JSON
        json_url = self.upload_json_to_azure(policy_json, payer_name, os.path.basename(pdf_path))

        # Dedup engine
        policy_id = None
        if self.dedup_engine:
            try:
                metadata = self.dedup_engine.process_extracted_policy(
                    content=policy_json,
                    raw_text=text,
                    source_pdf=os.path.basename(pdf_path),
                    source_url=source_url,
                    payer_name=payer_name
                )
                self.dedup_engine.remove_replaced_policies(metadata)
                self.dedup_engine.save_to_azure(metadata)
                policy_id = metadata.policy_id
            except Exception as e:
                self.logger.warning(f"Dedup pipeline failed for {pdf_path}: {e}")

        return {"json_url": json_url, "policy_id": policy_id}
    
    def is_healthcare_link_relevant(self, text, href):
        """Check if link is relevant for our target healthcare rules"""
        text_lower = text.lower()
        href_lower = href.lower()
        
        # Collect all keywords from target rules
        all_keywords = []
        for rule_data in self.target_rules.values():
            all_keywords.extend(rule_data['keywords'])
            all_keywords.extend(rule_data['url_patterns'])
        all_keywords.extend(self.extra_link_terms)
        
        combined = f"{text_lower} {href_lower}"
        return any(keyword in combined for keyword in all_keywords)
    
    def download_targeted_pdf(self, url, company_name, rule_classification, filename=None):
        """Download PDF that matches our target rules and upload to Azure if configured"""
        try:
            if not filename:
                filename = os.path.basename(urlparse(url).path)
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
            
            # Create rule-specific directory
            primary_rule = rule_classification['primary_rule']
            company_dir = Path(f"./targeted_pdfs/{company_name}/{primary_rule}")
            company_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = company_dir / filename
            
            # Skip if already exists
            if filepath.exists():
                self.logger.info(f"PDF already exists: {filepath}")
                return {"local_path": str(filepath), "azure_url": None}
            
            # Download
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Verify it's a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
                self.logger.warning(f"URL doesn't appear to be PDF: {url}")
                return None
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = filepath.stat().st_size
            self.logger.info(f"✅ Downloaded {primary_rule} PDF: {filename} ({file_size:,} bytes)")
            
            # Compute hash for dedupe check
            file_bytes = filepath.read_bytes()
            content_hash = hashlib.md5(file_bytes).hexdigest()
            
            # Upload to Azure if available and not already present
            azure_url = None
            if self.azure_uploader:
                existing_hashes = self._load_existing_hashes(company_name)
                if content_hash in existing_hashes:
                    self.logger.info(f"☁️ Skipping Azure upload (already present): {filename}")
                else:
                    try:
                        azure_url = self.azure_uploader.upload_pdf_from_file(
                            str(filepath),
                            payer_name=company_name,
                            metadata={
                                "primary_rule": primary_rule,
                                "crawl_date": time.strftime("%Y-%m-%d"),
                                "source_url": url,
                                "content_hash": content_hash
                            }
                        )
                        self.logger.info(f"☁️ Uploaded to Azure: {azure_url}")
                        existing_hashes.add(content_hash)
                    except Exception as upload_error:
                        self.logger.warning(f"Azure upload failed for {filename}: {upload_error}")
            
            # Convert to JSON and upload policy metadata/JSON
            json_info = self.convert_pdf_to_policy_json(
                pdf_path=str(filepath),
                payer_name=company_name,
                source_url=url,
                primary_rule=primary_rule
            )

            return {
                "local_path": str(filepath),
                "azure_url": azure_url,
                "json_url": json_info.get("json_url") if json_info else None,
                "policy_id": json_info.get("policy_id") if json_info else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to download PDF {url}: {e}")
            return None
    
    def discover_targeted_pdfs(self, payer_config):
        """Discover PDFs specifically for target rule types"""
        company_name = payer_config['name']
        targeted_urls = payer_config['targeted_urls']
        allowed_domains = payer_config['allowed_domains']
        
        self.logger.info(f"🎯 Starting targeted discovery for {company_name}")
        
        # Initialize queue with targeted URLs
        queue = deque([(url, 0) for url in targeted_urls])
        
        all_links_found = []
        targeted_pdfs = []
        
        # Reset visited URLs for each company
        self.visited_urls = set()
        
        while queue and len(self.visited_urls) < 25:  # Focused crawling
            current_url, depth = queue.popleft()
            
            if current_url in self.visited_urls or depth > self.max_depth:
                continue
            
            self.visited_urls.add(current_url)
            self.logger.info(f"Exploring {company_name} (depth {depth}): {current_url}")
            
            try:
                self.driver.get(current_url)
                time.sleep(2)
                
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                links = soup.find_all('a', href=True)
                
                for link in links:
                    href = link['href']
                    text = link.get_text(strip=True)
                    
                    absolute_url = urljoin(current_url, href)
                    parsed = urlparse(absolute_url)
                    
                    # Check domain
                    if not any(domain in parsed.netloc for domain in allowed_domains):
                        continue
                    
                    # Check if PDF
                    if self.is_pdf_url(absolute_url):
                        # Classify PDF relevance
                        classification = self.classify_pdf_relevance(text, href, absolute_url)
                        
                        if classification['is_targeted']:
                            self.logger.info(f"🎯 Found targeted PDF: {absolute_url}")
                            self.logger.info(f"   Primary rule: {classification['primary_rule']}")
                            self.logger.info(f"   All relevant rules: {classification['relevant_rules']}")
                            
                            # Download the PDF
                            downloaded_info = self.download_targeted_pdf(
                                absolute_url, company_name, classification
                            )
                            
                            if downloaded_info:
                                targeted_pdfs.append({
                                    'url': absolute_url,
                                    'local_path': downloaded_info.get('local_path'),
                                    'azure_url': downloaded_info.get('azure_url'),
                                    'json_url': downloaded_info.get('json_url'),
                                    'policy_id': downloaded_info.get('policy_id'),
                                    'rule_classification': classification,
                                    'discovered_at_depth': depth,
                                    'parent_url': current_url,
                                    'link_text': text
                                })
                    
                    # Add relevant links for further exploration
                    elif (self.is_healthcare_link_relevant(text, href) and 
                          depth < self.max_depth and 
                          absolute_url not in self.visited_urls):
                        queue.append((absolute_url, depth + 1))
                        
                        all_links_found.append({
                            'url': absolute_url,
                            'text': text,
                            'depth': depth
                        })
                
            except Exception as e:
                self.logger.warning(f"Error processing {current_url}: {e}")
                continue
        
        return {
            'company_name': company_name,
            'targeted_pdfs': targeted_pdfs,
            'all_links': all_links_found,
            'total_targeted_pdfs': len(targeted_pdfs),
            'urls_visited': len(self.visited_urls)
        }
    
    def crawl_targeted_companies(self, selected_companies=None):
        """Crawl for targeted healthcare rule PDFs"""
        payer_configs = self.load_payer_configurations()
        
        if selected_companies:
            payer_configs = {k: v for k, v in payer_configs.items() if k in selected_companies}
        
        print(f"🎯 Targeted Healthcare Rule Crawler")
        print(f"Target Rules: {', '.join(self.target_rules.keys())}")
        print(f"Companies: {len(payer_configs)}")
        print("=" * 70)
        
        all_results = []
        total_targeted_pdfs = 0
        pdfs_by_rule_type = {rule: 0 for rule in self.target_rules.keys()}
        
        for company_key, payer_config in payer_configs.items():
            print(f"\\n🎯 Processing: {payer_config['name']}")
            print("-" * 50)
            
            try:
                results = self.discover_targeted_pdfs(payer_config)
                all_results.append(results)
                
                total_targeted_pdfs += results['total_targeted_pdfs']
                
                # Count PDFs by rule type
                for pdf_info in results['targeted_pdfs']:
                    primary_rule = pdf_info['rule_classification']['primary_rule']
                    pdfs_by_rule_type[primary_rule] += 1
                
                print(f"   ✅ {results['total_targeted_pdfs']} targeted PDFs downloaded")
                print(f"   🌐 {results['urls_visited']} URLs visited")
                
                # Show rule type breakdown for this company
                company_rule_counts = {}
                for pdf_info in results['targeted_pdfs']:
                    rule = pdf_info['rule_classification']['primary_rule']
                    company_rule_counts[rule] = company_rule_counts.get(rule, 0) + 1
                
                for rule, count in company_rule_counts.items():
                    print(f"   📋 {rule.replace('_', ' ').title()}: {count} PDFs")
                
                time.sleep(3)  # Respectful delay
                
            except Exception as e:
                self.logger.error(f"Failed processing {payer_config['name']}: {e}")
                continue
        
        # Update processing log
        self.processing_log.update({
            'companies_processed': all_results,
            'total_targeted_pdfs': total_targeted_pdfs,
            'pdfs_by_rule_type': pdfs_by_rule_type
        })
        
        # Save log
        with open('targeted_crawl_log.json', 'w') as f:
            json.dump(self.processing_log, f, indent=2)
        
        # Display final summary
        print(f"\\n🎉 Targeted Crawling Complete!")
        print("=" * 50)
        print(f"📊 Total targeted PDFs: {total_targeted_pdfs}")
        print(f"📋 Rule Type Breakdown:")
        for rule_type, count in pdfs_by_rule_type.items():
            print(f"   • {rule_type.replace('_', ' ').title()}: {count} PDFs")
        
        print(f"\\n📁 PDFs organized in: ./targeted_pdfs/[company]/[rule_type]/")
        print(f"📄 Log saved to: targeted_crawl_log.json")
        
        return self.processing_log

def main():
    """Main function for targeted healthcare rule crawling"""
    print("🎯 Targeted Healthcare Rule Crawler")
    print("Focusing on: Timely Filing, Prior Authorization, Billing, Appeals")
    print("=" * 70)
    
    crawler = TargetedHealthcareRuleCrawler(headless=True, max_depth=2)
    
    try:
        # Focus on major payers first
        results = crawler.crawl_targeted_companies([
            'united_healthcare', 'anthem', 'aetna', 'humana', 'cigna', 'countycare'
        ])
        
        print(f"\\n✅ Targeted crawling completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Targeted crawling failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        crawler.close()

if __name__ == "__main__":
    main()
