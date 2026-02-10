"""
Integration module for BIG_KnowledgeBase project
Connects PDF crawler → OCR/Extraction → Deduplication → Azure Storage
"""

import os
import json
import re
import textwrap
from typing import List, Dict
from datetime import datetime
from azure.storage.blob import BlobServiceClient
import PyPDF2
import fitz  # PyMuPDF
from huggingface_hub import InferenceClient

# Import your existing crawler (adjust path as needed)
# from payer_portal_crawler import PayerPortalCrawler
# from intelligent_csv_crawler import IntelligentCSVCrawler

# Import the deduplication system
from .policy_deduplication_system import PolicyDeduplicationEngine, PolicyMetadata


class PDFToStructuredPipeline:
    """
    Complete pipeline: PDF → OCR → JSON → Deduplicated Storage
    """
    
    def __init__(self, 
                 azure_connection_string: str,
                 temp_pdf_dir: str = "./temp_pdfs",
                 temp_json_dir: str = "./temp_json"):
        
        self.azure_connection = azure_connection_string
        self.blob_service = BlobServiceClient.from_connection_string(azure_connection_string)
        self.dedup_engine = PolicyDeduplicationEngine(azure_connection_string)
        
        self.temp_pdf_dir = temp_pdf_dir
        self.temp_json_dir = temp_json_dir
        
        # Create temp directories
        os.makedirs(temp_pdf_dir, exist_ok=True)
        os.makedirs(temp_json_dir, exist_ok=True)
        
        # Container for raw PDFs
        self.pdf_container = "raw-pdfs"
        self._ensure_container(self.pdf_container)
        # Container for JSON outputs
        self.json_container = os.getenv("AZURE_JSON_CONTAINER", "policy-json")
        self._ensure_container(self.json_container)

        # Hugging Face client for policy JSON extraction
        self.hf_model = os.getenv("HF_POLICY_MODEL", "microsoft/Phi-3.5-mini-instruct")
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.hf_client = None
        if hf_token:
            try:
                self.hf_client = InferenceClient(model=self.hf_model, token=hf_token)
                print(f"Hugging Face client initialized: {self.hf_model}")
            except Exception as e:
                print(f"Warning: could not initialize HF client: {e}")
    
    def _ensure_container(self, container_name: str):
        """Ensure Azure container exists"""
        try:
            self.blob_service.create_container(container_name)
        except Exception:
            pass

    def generate_policy_json_with_hf(self, text: str, payer_name: str, source_pdf: str) -> Dict:
        """Use HF model to extract structured policy JSON"""
        if not self.hf_client:
            return {}

        prompt = textwrap.dedent(f"""
        You are a healthcare policy extractor. Read the policy text and return ONLY compact JSON with these keys:
        policy_id (string), policy_type (prior_auth|timely_filing|appeals|claims|billing|coverage|unknown),
        effective_date (YYYY-MM-DD or null), end_date (YYYY-MM-DD or null),
        supersedes (array of strings), summary (string), payer_name (string),
        source_pdf (string).

        Payer: "{payer_name}"
        Source PDF: "{source_pdf}"

        Policy text (may be truncated):
        {text[:6000]}

        JSON only, no prose, no code fences.
        """).strip()

        try:
            resp = self.hf_client.text_generation(
                prompt=prompt,
                max_new_tokens=600,
                temperature=0.2,
                do_sample=False,
                stop=["</s>", "```"]
            )
            cleaned = resp.strip().strip("`")
            return json.loads(cleaned)
        except Exception as e:
            print(f"HF extraction failed: {e}")
            return {}

    def upload_json_to_azure(self, data: Dict, payer_name: str, source_pdf: str) -> str:
        """Upload extracted JSON to dedicated container"""
        try:
            blob_name = f"{payer_name}/{os.path.splitext(source_pdf)[0]}.json"
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
            print(f"Failed to upload JSON for {source_pdf}: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path: str) -> tuple[str, Dict]:
        """
        Extract text from PDF using dual method (PyMuPDF + PyPDF2)
        Similar to your existing PDF processor
        """
        text_content = ""
        metadata = {}
        
        # Method 1: PyMuPDF (better for most PDFs)
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text_content += page.get_text()
            
            metadata = {
                'page_count': len(doc),
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'creation_date': doc.metadata.get('creationDate', '')
            }
            doc.close()
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
        
        # Method 2: PyPDF2 fallback
        if not text_content:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text_content += page.extract_text()
                    
                    metadata['page_count'] = len(pdf_reader.pages)
            except Exception as e:
                print(f"PyPDF2 extraction failed: {e}")
        
        return text_content, metadata
    
    def extract_rules_from_text(self, text: str, payer_name: str) -> List[Dict]:
        """
        Extract healthcare rules from text using regex patterns
        Similar to your existing rule extraction engine
        """
        rules = []
        
        # Healthcare-specific patterns (from your existing code)
        patterns = {
            'prior_authorization': [
                r'prior authorization.{0,100}required',
                r'PA.{0,50}required',
                r'pre-authorization.{0,100}necessary'
            ],
            'timely_filing': [
                r'timely filing.{0,100}\d+\s*days',
                r'claim.{0,50}submit.{0,50}within.{0,50}\d+\s*days',
                r'filing deadline.{0,100}\d+\s*days'
            ],
            'appeals': [
                r'appeal.{0,100}\d+\s*days',
                r'reconsideration.{0,100}\d+\s*days',
                r'dispute.{0,100}within.{0,50}\d+\s*days'
            ],
            'claims': [
                r'claim submission.{0,100}',
                r'billing requirements.{0,100}',
                r'reimbursement.{0,100}'
            ]
        }
        
        # Extract rules by category
        for category, category_patterns in patterns.items():
            for pattern in category_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Get context around the match
                    start = max(0, match.start() - 200)
                    end = min(len(text), match.end() + 200)
                    context = text[start:end].strip()
                    
                    rule = {
                        'policy_type': category,
                        'payer_name': payer_name,
                        'matched_text': match.group(0),
                        'context': context,
                        'extracted_at': datetime.now().isoformat()
                    }
                    rules.append(rule)
        
        return rules
    
    def process_single_pdf(self, 
                          pdf_path: str, 
                          payer_name: str,
                          source_url: str = "") -> List[PolicyMetadata]:
        """
        Complete processing for a single PDF:
        1. Extract text (OCR)
        2. Extract rules (Pattern matching)
        3. Create policy metadata (Hybrid ID extraction)
        4. Save to Azure
        """
        
        print(f"Processing PDF: {pdf_path}")
        
        # Step 1: Extract text from PDF
        raw_text, pdf_metadata = self.extract_text_from_pdf(pdf_path)
        
        if not raw_text:
            print(f"No text extracted from {pdf_path}")
            return []
        
        # Step 2: Try HF policy JSON extraction
        policy_json = self.generate_policy_json_with_hf(
            text=raw_text,
            payer_name=payer_name,
            source_pdf=os.path.basename(pdf_path)
        )

        if not policy_json:
            # Fallback to regex rules if HF failed
            rules = self.extract_rules_from_text(raw_text, payer_name)
            if not rules:
                print(f"No rules found in {pdf_path}")
                return []
            # Use first rule as content for metadata
            policy_json = rules[0]

        # Upload JSON representation
        json_url = self.upload_json_to_azure(policy_json, payer_name, os.path.basename(pdf_path))
        if json_url:
            print(f"Uploaded policy JSON to Azure: {json_url}")

        # Step 3: Process through deduplication engine
        policies = []
        try:
            policy_metadata = self.dedup_engine.process_extracted_policy(
                content=policy_json,
                raw_text=raw_text,
                source_pdf=os.path.basename(pdf_path),
                source_url=source_url,
                payer_name=payer_name
            )
            
            # Step 4: Remove replaced/expired versions, then save to Azure
            try:
                self.dedup_engine.remove_replaced_policies(policy_metadata)
            except Exception as e:
                print(f"Warning: could not remove replaced policies: {e}")
            self.dedup_engine.save_to_azure(policy_metadata)
            policies.append(policy_metadata)
            
        except Exception as e:
            print(f"Error processing policy: {e}")
        
        print(f"Created {len(policies)} policy metadata objects")
        
        return policies
    
    def process_pdf_batch(self, 
                         pdf_paths: List[str], 
                         payer_name: str,
                         source_urls: List[str] = None) -> Dict:
        """
        Process multiple PDFs in batch
        """
        
        if source_urls is None:
            source_urls = [""] * len(pdf_paths)
        
        all_policies = []
        stats = {
            'total_pdfs': len(pdf_paths),
            'successful': 0,
            'failed': 0,
            'total_policies': 0,
            'start_time': datetime.now().isoformat()
        }
        
        for pdf_path, source_url in zip(pdf_paths, source_urls):
            try:
                policies = self.process_single_pdf(pdf_path, payer_name, source_url)
                all_policies.extend(policies)
                stats['successful'] += 1
                stats['total_policies'] += len(policies)
            except Exception as e:
                print(f"Failed to process {pdf_path}: {e}")
                stats['failed'] += 1
        
        stats['end_time'] = datetime.now().isoformat()
        
        # After batch processing, run deduplication
        print("\n=== Running deduplication across all policies ===")
        cleanup_stats = self.dedup_engine.cleanup_expired_policies()
        
        stats['cleanup'] = cleanup_stats
        
        return stats


class CrawlerIntegration:
    """
    Integration with your existing crawler systems
    """
    
    def __init__(self, azure_connection_string: str):
        self.pipeline = PDFToStructuredPipeline(azure_connection_string)
    
    def integrate_with_basic_crawler(self, payer_name: str = "anthem"):
        """
        Integration example with basic payer_portal_crawler.py
        """
        # Your existing crawler code
        # from payer_portal_crawler import PayerPortalCrawler
        # crawler = PayerPortalCrawler()
        # results = crawler.crawl_payer(payer_name)
        
        # Simulate results
        results = {
            'pdfs_downloaded': [
                f'/path/to/{payer_name}_policy_1.pdf',
                f'/path/to/{payer_name}_policy_2.pdf'
            ],
            'source_urls': [
                f'https://{payer_name}.com/policy1',
                f'https://{payer_name}.com/policy2'
            ]
        }
        
        # Process all downloaded PDFs
        stats = self.pipeline.process_pdf_batch(
            pdf_paths=results['pdfs_downloaded'],
            payer_name=payer_name,
            source_urls=results['source_urls']
        )
        
        return stats
    
    def integrate_with_csv_crawler(self, csv_path: str = "payer_companies.csv"):
        """
        Integration example with intelligent_csv_crawler.py
        Process multiple payers from CSV
        """
        # from intelligent_csv_crawler import IntelligentCSVCrawler
        # crawler = IntelligentCSVCrawler(csv_path)
        
        # Simulate processing multiple payers
        payers = ['anthem', 'uhc', 'aetna', 'kaiser']
        
        all_stats = {}
        
        for payer in payers:
            print(f"\n{'='*60}")
            print(f"Processing payer: {payer}")
            print(f"{'='*60}")
            
            # Your crawler would download PDFs here
            # results = crawler.crawl_payer(payer)
            
            # Process downloaded PDFs
            stats = self.integrate_with_basic_crawler(payer)
            all_stats[payer] = stats
        
        return all_stats
    
    def integrate_with_bfs_crawler(self, 
                                   starting_urls: List[str],
                                   allowed_domains: List[str],
                                   payer_name: str):
        """
        Integration example with test_bfs_crawler.py
        """
        # from test_bfs_crawler import SimpleBFSCrawler
        # crawler = SimpleBFSCrawler(max_depth=3)
        # pdf_urls = crawler.discover_pdfs_bfs(starting_urls, allowed_domains)
        
        # Simulate discovered PDFs
        pdf_urls = [
            'https://example.com/policy1.pdf',
            'https://example.com/policy2.pdf'
        ]
        
        # Download PDFs (your existing logic)
        pdf_paths = []
        for url in pdf_urls:
            # Your download logic here
            # pdf_path = download_pdf(url)
            pdf_path = f"./temp/{payer_name}_{len(pdf_paths)}.pdf"
            pdf_paths.append(pdf_path)
        
        # Process all discovered PDFs
        stats = self.pipeline.process_pdf_batch(
            pdf_paths=pdf_paths,
            payer_name=payer_name,
            source_urls=pdf_urls
        )
        
        return stats


# Scheduled cleanup job
def scheduled_cleanup_job(azure_connection_string: str):
    """
    Run this as a daily/weekly scheduled job to clean up expired policies
    Can be deployed as Azure Function or cron job
    """
    print(f"Starting scheduled cleanup job at {datetime.now()}")
    
    engine = PolicyDeduplicationEngine(azure_connection_string)
    stats = engine.cleanup_expired_policies()
    
    # Save cleanup report
    report_blob = engine.blob_service.get_blob_client(
        container="healthcare-policies-metadata",
        blob=f"cleanup_reports/cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    report_blob.upload_blob(json.dumps(stats, indent=2))
    
    print("Cleanup job completed")
    return stats


# Example usage
if __name__ == "__main__":
    # Configuration
    AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    
    # Initialize integration
    integration = CrawlerIntegration(AZURE_CONNECTION_STRING)
    
    # Example 1: Process a single payer
    print("Example 1: Single payer processing")
    stats = integration.integrate_with_basic_crawler("anthem")
    print(json.dumps(stats, indent=2))
    
    # Example 2: Process multiple payers from CSV
    print("\n\nExample 2: Multi-payer CSV processing")
    all_stats = integration.integrate_with_csv_crawler()
    print(json.dumps(all_stats, indent=2))
    
    # Example 3: Run cleanup job
    print("\n\nExample 3: Running cleanup job")
    cleanup_stats = scheduled_cleanup_job(AZURE_CONNECTION_STRING)
    print(json.dumps(cleanup_stats, indent=2))
