#!/usr/bin/env python3
"""
Single Company Targeted Healthcare Crawler
Focused crawler for ONE healthcare company at a time to find specific rule PDFs:
- Timely Filing
- Prior Authorization  
- Billing Nuances
- Appeals

Usage:
    python single_company_targeted_crawler.py united_healthcare
    python single_company_targeted_crawler.py anthem
    python single_company_targeted_crawler.py aetna
"""

import time
import logging
import json
import os
import sys
import argparse
import hashlib
from collections import deque
from urllib.parse import urljoin, urlparse
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from dotenv import load_dotenv

from pipeline.azure_pdf_uploader import AzurePDFUploader

class SingleCompanyTargetedCrawler:
    """Crawl ONE healthcare company for targeted rule PDFs"""
    
    def __init__(self, headless=True, max_depth=3):
        self.headless = headless
        self.max_depth = max_depth
        self.visited_urls = set()
        self.discovered_pdfs = []
        self.extra_link_terms = [
            'policy', 'policies', 'bulletin', 'bulletins', 'manual', 'guide',
            'guidelines', 'criteria', 'protocol', 'coverage', 'medical policy',
            'clinical policy', 'provider manual'
        ]
        self.excluded_terms = [
            'genetic', 'molecular', 'lab test', 'laboratory', 'drug list',
            'formulary', 'immunization', 'vaccine', 'dental', 'oral health',
            'hearing aid', 'vision', 'prosthetic', 'sleep apnea', 'cpap',
            'clinical policy bulletin', 'medical policy'
        ]
        
        # Target rule types with healthcare-specific keywords
        self.target_rules = {
            'timely_filing': {
                'keywords': [
                    'timely filing', 'filing deadline', 'filing requirement', 'filing limit',
                    'claim submission deadline', 'submission timeframe', 'filing window',
                    'claim filing', 'timely submission', 'deadline requirement',
                    '90 days', '180 days', '365 days', 'submission deadline'
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
                    'prior approval', 'authorization request', 'coverage approval',
                    'advance notification'
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
                    'modifier', 'cpt code', 'icd code', 'procedure code',
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
                    'appeal rights', 'complaint resolution', 'review process',
                    'first level appeal', 'second level appeal'
                ],
                'url_patterns': [
                    'appeal', 'grievance', 'dispute', 'complaint', 'review'
                ]
            }
        }
        
        self.processing_log = {
            "crawl_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_rules": list(self.target_rules.keys()),
            "company_processed": None,
            "total_targeted_pdfs": 0,
            "pdfs_by_rule_type": {}
        }
        
        self.setup_webdriver()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )
        self.logger = logging.getLogger(__name__)
        
        # Azure uploader setup
        load_dotenv()
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container_name = os.getenv("AZURE_CONTAINER_NAME", "pdfs")
        self.azure_uploader = None
        self._existing_hashes_cache = {}
        if connection_string:
            try:
                self.azure_uploader = AzurePDFUploader(connection_string, container_name)
                self.logger.info(f"Azure uploader configured for container '{container_name}'")
            except Exception as e:
                self.logger.warning(f"Azure uploader not initialized: {e}")
    
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
    
    def get_company_config(self, company_name):
        """Get configuration for specific healthcare company"""
        configs = {
            "united_healthcare": {
                "name": "United Healthcare",
                "base_url": "https://www.uhcprovider.com/",
                "targeted_urls": [
                    "https://www.uhcprovider.com/en/prior-authorization.html",
                    "https://www.uhcprovider.com/en/claims-payments.html", 
                    "https://www.uhcprovider.com/en/policies-protocols.html",
                    "https://www.uhcprovider.com/en/admin-guides.html"
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
            "kaiser": {
                "name": "Kaiser Permanente",
                "base_url": "https://provider.kaiserpermanente.org/",
                "targeted_urls": [
                    "https://provider.kaiserpermanente.org/",
                    "https://provider.kaiserpermanente.org/static/pdf/"
                ],
                "allowed_domains": ["kaiserpermanente.org", "provider.kaiserpermanente.org"]
            },
            "molina": {
                "name": "Molina Healthcare",
                "base_url": "https://www.molinahealthcare.com/providers/",
                "targeted_urls": [
                    "https://www.molinahealthcare.com/providers/medicaid",
                    "https://www.molinahealthcare.com/providers/medicare"
                ],
                "allowed_domains": ["molinahealthcare.com"]
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
            },
            "bcbs_illinois": {
                "name": "BCBS of Illinois",
                "base_url": "https://www.bcbsil.com/provider/",
                "targeted_urls": [
                    "https://www.bcbsil.com/provider/",
                    "https://www.bcbsil.com/provider/claims"
                ],
                "allowed_domains": ["bcbsil.com"]
            },
            "florida_blue": {
                "name": "Florida Blue",
                "base_url": "https://www.floridablue.com/providers/",
                "targeted_urls": [
                    "https://www.floridablue.com/providers/claims-and-reimbursement",
                    "https://www.floridablue.com/providers/prior-authorization"
                ],
                "allowed_domains": ["floridablue.com"]
            }
        }
        
        return configs.get(company_name.lower())
    
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
            scores['billing_nuances'] = 2  # meet minimum threshold
            relevant_rules.append('billing_nuances')
        
        # Determine primary rule type (highest score)
        primary_rule = max(scores.keys(), key=lambda x: scores[x]) if scores else None

        # Require a minimum score to avoid weak matches
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
    
    def download_pdf(self, url, company_name, rule_classification, filename=None):
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
            
            return {"local_path": str(filepath), "azure_url": azure_url}
            
        except Exception as e:
            self.logger.error(f"Failed to download PDF {url}: {e}")
            return None
    
    def discover_targeted_pdfs(self, company_config):
        """Discover PDFs specifically for target rule types from ONE company"""
        company_name = company_config['name']
        targeted_urls = company_config['targeted_urls']
        allowed_domains = company_config['allowed_domains']
        
        self.logger.info(f"🎯 Starting targeted discovery for {company_name}")
        print(f"🏥 Crawling: {company_name}")
        print(f"🎯 Target Rules: {', '.join(self.target_rules.keys())}")
        print(f"📋 Starting URLs: {len(targeted_urls)}")
        print("-" * 60)
        
        # Initialize queue with targeted URLs
        queue = deque([(url, 0) for url in targeted_urls])
        
        all_links_found = []
        targeted_pdfs = []
        
        # Reset visited URLs
        self.visited_urls = set()
        
        while queue and len(self.visited_urls) < 50:  # Focused crawling - limit per company
            current_url, depth = queue.popleft()
            
            if current_url in self.visited_urls or depth > self.max_depth:
                continue
            
            self.visited_urls.add(current_url)
            self.logger.info(f"Exploring depth {depth}: {current_url}")
            
            try:
                self.driver.get(current_url)
                time.sleep(2)  # Be respectful
                
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
                            print(f"🎯 Found targeted PDF: {os.path.basename(absolute_url)}")
                            print(f"   Primary rule: {classification['primary_rule']}")
                            print(f"   All relevant rules: {classification['relevant_rules']}")
                            print(f"   Link text: {text[:50]}")
                            
                            # Download the PDF
                            downloaded_info = self.download_pdf(
                                absolute_url, company_name, classification
                            )
                            
                            if downloaded_info:
                                targeted_pdfs.append({
                                    'url': absolute_url,
                                    'local_path': downloaded_info.get('local_path'),
                                    'azure_url': downloaded_info.get('azure_url'),
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
    
    def crawl_single_company(self, company_name):
        """Crawl one specific healthcare company"""
        # Get company configuration
        company_config = self.get_company_config(company_name)
        
        if not company_config:
            available_companies = [
                'united_healthcare', 'anthem', 'aetna', 'humana', 'cigna', 
                'kaiser', 'molina', 'countycare', 'bcbs_illinois', 'florida_blue'
            ]
            print(f"❌ Unknown company: {company_name}")
            print(f"Available companies: {', '.join(available_companies)}")
            return None
        
        print(f"🎯 Single Company Targeted Healthcare Crawler")
        print(f"Company: {company_config['name']}")
        print(f"Target Rules: {', '.join(self.target_rules.keys())}")
        print("=" * 70)
        
        try:
            # Run targeted discovery
            results = self.discover_targeted_pdfs(company_config)
            
            # Count PDFs by rule type
            pdfs_by_rule_type = {}
            for pdf_info in results['targeted_pdfs']:
                primary_rule = pdf_info['rule_classification']['primary_rule']
                pdfs_by_rule_type[primary_rule] = pdfs_by_rule_type.get(primary_rule, 0) + 1
            
            # Update processing log
            self.processing_log.update({
                'company_processed': results['company_name'],
                'total_targeted_pdfs': results['total_targeted_pdfs'],
                'pdfs_by_rule_type': pdfs_by_rule_type,
                'results': results
            })
            
            # Save log
            log_filename = f"{company_name}_crawl_log.json"
            with open(log_filename, 'w') as f:
                json.dump(self.processing_log, f, indent=2)
            
            # Display results
            print(f"\\n🎉 Crawling Complete for {results['company_name']}!")
            print("=" * 50)
            print(f"📊 Total targeted PDFs: {results['total_targeted_pdfs']}")
            print(f"🌐 URLs visited: {results['urls_visited']}")
            
            if pdfs_by_rule_type:
                print(f"\\n📋 PDFs by Rule Type:")
                for rule_type, count in pdfs_by_rule_type.items():
                    print(f"   • {rule_type.replace('_', ' ').title()}: {count} PDFs")
            
            print(f"\\n📁 PDFs downloaded to: ./targeted_pdfs/{results['company_name']}/")
            print(f"📄 Log saved to: {log_filename}")
            
            return self.processing_log
            
        except Exception as e:
            self.logger.error(f"Failed crawling {company_config['name']}: {e}")
            print(f"❌ Crawling failed: {e}")
            return None

def main():
    """Main function for single company targeted crawling"""
    parser = argparse.ArgumentParser(
        description="Single Company Targeted Healthcare Rule Crawler"
    )
    parser.add_argument(
        "company", 
        help="Company to crawl",
        choices=[
            'united_healthcare', 'anthem', 'aetna', 'humana', 'cigna',
            'kaiser', 'molina', 'countycare', 'bcbs_illinois', 'florida_blue'
        ]
    )
    parser.add_argument(
        "--headless", 
        action="store_true", 
        default=True,
        help="Run in headless mode (default: True)"
    )
    parser.add_argument(
        "--max-depth", 
        type=int, 
        default=3,
        help="Maximum crawl depth (default: 3)"
    )
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        # Show usage if no arguments
        print("🎯 Single Company Targeted Healthcare Crawler")
        print("=" * 50)
        print("Usage:")
        print("  python single_company_targeted_crawler.py <company_name>")
        print("")
        print("Available companies:")
        companies = [
            'united_healthcare', 'anthem', 'aetna', 'humana', 'cigna',
            'kaiser', 'molina', 'countycare', 'bcbs_illinois', 'florida_blue'
        ]
        for company in companies:
            print(f"  - {company}")
        print("")
        print("Example:")
        print("  python single_company_targeted_crawler.py united_healthcare")
        return
    
    crawler = SingleCompanyTargetedCrawler(
        headless=args.headless, 
        max_depth=args.max_depth
    )
    
    try:
        results = crawler.crawl_single_company(args.company)
        
        if results:
            print(f"\\n✅ Success! Found {results['total_targeted_pdfs']} targeted PDFs")
        else:
            print(f"\\n❌ Crawling failed")
            
        return results is not None
        
    except KeyboardInterrupt:
        print("\\n⏹️  Crawling interrupted by user")
        return False
    except Exception as e:
        print(f"\\n❌ Crawling failed: {e}")
        return False
    finally:
        crawler.close()

if __name__ == "__main__":
    main()
