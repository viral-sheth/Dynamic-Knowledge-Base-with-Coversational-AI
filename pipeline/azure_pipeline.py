#!/usr/bin/env python3
"""
Azure Pipeline with .env file support
"""

import os
import re
import json
import shutil
from datetime import datetime
from typing import List, Dict, Optional

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads your .env file!
    print("✓ Loaded .env file")
except ImportError:
    print("⚠ python-dotenv not installed, trying environment variables only")

# Azure imports
from azure.storage.blob import BlobServiceClient

# PDF processing imports
import PyPDF2
import fitz  # PyMuPDF


class SimplePDFProcessor:
    """Simple PDF processor without Document Intelligence"""
    
    def __init__(self):
        self.patterns = {
            'prior_auth': r'prior\s+auth|pre[-\s]?auth|PA\s+required',
            'timely_filing': r'timely\s+filing|filing\s+limit|submission\s+deadline',
            'appeal': r'appeal|reconsideration|dispute|grievance',
            'claim': r'claim\s+submission|billing|reimbursement',
            'coverage': r'coverage|medical\s+necessity',
            'network': r'in[-\s]?network|out[-\s]?of[-\s]?network'
        }
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF first, fallback to PyPDF2"""
        text = ""
        
        # Try PyMuPDF
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            doc.close()
            
            if len(text.strip()) > 100:
                return text
        except Exception as e:
            print(f"    PyMuPDF failed: {str(e)[:50]}")
        
        # Fallback to PyPDF2
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            print(f"    PyPDF2 failed: {str(e)[:50]}")
            return ""
    
    def extract_rules(self, text: str) -> List[Dict]:
        """Extract rules using regex patterns"""
        rules = []
        sentences = re.split(r'[.!?]\n', text)
        
        for sentence in sentences:
            if len(sentence) < 20:
                continue
            
            for rule_type, pattern in self.patterns.items():
                if re.search(pattern, sentence, re.IGNORECASE):
                    rules.append({
                        'type': rule_type,
                        'text': sentence.strip()[:500],
                        'confidence': 0.8
                    })
                    break
        
        return rules
    
    def process(self, pdf_path: str) -> Optional[Dict]:
        """Process a single PDF"""
        filename = os.path.basename(pdf_path)
        print(f"\n  Processing: {filename}")
        
        try:
            text = self.extract_text(pdf_path)
            
            if not text or len(text.strip()) < 100:
                print(f"    ✗ No text extracted")
                return None
            
            print(f"    ✓ Extracted {len(text)} characters")
            
            rules = self.extract_rules(text)
            print(f"    ✓ Found {len(rules)} rules")
            
            if len(rules) == 0:
                print(f"    ⚠ No rules found - skipping")
                return None
            
            policy = {
                'metadata': {
                    'filename': filename,
                    'file_size': os.path.getsize(pdf_path),
                    'extraction_date': datetime.now().isoformat(),
                    'character_count': len(text),
                    'payer': self._extract_payer(filename, text)
                },
                'rules': rules,
                'rule_count': len(rules),
                'sample_text': text[:2000],
                'status': 'success'
            }
            
            return policy
            
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            return None
    
    def _extract_payer(self, filename: str, text: str) -> str:
        """Identify payer from filename or content"""
        filename_lower = filename.lower()
        text_sample = text[:3000].lower()
        
        payers = {
            'UHC': ['united', 'uhc', 'unitedhealthcare'],
            'ANTHEM': ['anthem', 'elevance'],
            'AETNA': ['aetna', 'cvs'],
            'KAISER': ['kaiser'],
            'HUMANA': ['humana'],
            'CIGNA': ['cigna']
        }
        
        for payer, keywords in payers.items():
            if any(kw in filename_lower for kw in keywords):
                return payer
            if any(kw in text_sample for kw in keywords):
                return payer
        
        return 'UNKNOWN'


class SimpleAzurePipeline:
    """Simplified Azure pipeline with .env support"""
    
    def __init__(self, connection_string: str):
        self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        self.processor = SimplePDFProcessor()
        
        self.containers = {
            'pdfs': 'pdfs',
            'policies': 'healthcare-policies',
            'metadata': 'healthcare-policies-metadata',
            'archive': 'archived-policies'
        }
    
    def download_pdfs(self, output_dir: str = "./temp_pdfs/") -> List[str]:
        """Download PDFs from Azure"""
        print("\n" + "="*60)
        print("STEP 1: DOWNLOADING PDFs FROM AZURE")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        downloaded = []
        
        try:
            container = self.blob_service.get_container_client(self.containers['pdfs'])
            blobs = container.list_blobs()
            
            for blob in blobs:
                if not blob.name.endswith('.pdf'):
                    continue
                
                # Handle nested paths (e.g., anthem/2025-11/file.pdf)
                local_path = os.path.join(output_dir, blob.name)
                
                # Create nested directories if needed
                local_dir = os.path.dirname(local_path)
                os.makedirs(local_dir, exist_ok=True)
                
                try:
                    blob_client = container.get_blob_client(blob.name)
                    with open(local_path, 'wb') as f:
                        f.write(blob_client.download_blob().readall())
                    
                    downloaded.append(local_path)
                    print(f"  ✓ {blob.name}")
                    
                except Exception as e:
                    print(f"  ✗ {blob.name}: {str(e)[:50]}")
            
            print(f"\n  Total downloaded: {len(downloaded)} PDFs")
            return downloaded
            
        except Exception as e:
            print(f"  ✗ Error accessing Azure: {str(e)}")
            return []
    
    def process_pdfs(self, pdf_paths: List[str]) -> List[Dict]:
        """Process all PDFs"""
        print("\n" + "="*60)
        print("STEP 2: PROCESSING PDFs")
        print("="*60)
        
        policies = []
        
        for pdf_path in pdf_paths:
            policy = self.processor.process(pdf_path)
            if policy:
                policies.append(policy)
        
        print(f"\n  Successfully processed: {len(policies)}/{len(pdf_paths)} PDFs")
        print(f"  Total rules extracted: {sum(p['rule_count'] for p in policies)}")
        
        return policies
    
    def upload_policies(self, policies: List[Dict]) -> int:
        """Upload policies to Azure"""
        print("\n" + "="*60)
        print("STEP 3: UPLOADING TO AZURE")
        print("="*60)
        
        container = self.blob_service.get_container_client(self.containers['policies'])
        uploaded = 0
        
        for policy in policies:
            try:
                blob_name = policy['metadata']['filename'].replace('.pdf', '.json')
                blob_client = container.get_blob_client(blob_name)
                
                policy_json = json.dumps(policy, indent=2)
                blob_client.upload_blob(policy_json, overwrite=True)
                
                uploaded += 1
                print(f"  ✓ {blob_name}")
                
            except Exception as e:
                print(f"  ✗ {blob_name}: {str(e)[:50]}")
        
        print(f"\n  Uploaded: {uploaded} policies")
        return uploaded
    
    def create_index(self, policies: List[Dict]):
        """Create and upload master index"""
        print("\n" + "="*60)
        print("STEP 4: CREATING INDEX")
        print("="*60)
        
        index = {
            'last_updated': datetime.now().isoformat(),
            'total_policies': len(policies),
            'total_rules': sum(p['rule_count'] for p in policies),
            'policies': [
                {
                    'filename': p['metadata']['filename'],
                    'payer': p['metadata']['payer'],
                    'rule_count': p['rule_count'],
                    'extraction_date': p['metadata']['extraction_date']
                }
                for p in policies
            ]
        }
        
        try:
            container = self.blob_service.get_container_client(self.containers['metadata'])
            blob = container.get_blob_client('policy_index.json')
            blob.upload_blob(json.dumps(index, indent=2), overwrite=True)
            print("  ✓ Index uploaded")
        except Exception as e:
            print(f"  ✗ Index upload failed: {str(e)}")
    
    def cleanup(self, temp_dir: str = "./temp_pdfs/"):
        """Remove temporary files"""
        print("\n" + "="*60)
        print("STEP 5: CLEANUP")
        print("="*60)
        
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                print("  ✓ Temporary files deleted")
            else:
                print("  ○ No temporary files to clean")
        except Exception as e:
            print(f"  ✗ Cleanup failed: {str(e)}")
    
    def run(self):
        """Execute complete pipeline"""
        print("\n" + "="*60)
        print("HEALTHCARE POLICY PIPELINE")
        print("="*60)
        
        start_time = datetime.now()
        
        pdf_paths = self.download_pdfs()
        if not pdf_paths:
            print("\n⚠ No PDFs found!")
            return
        
        policies = self.process_pdfs(pdf_paths)
        if not policies:
            print("\n⚠ No policies extracted!")
            return
        
        uploaded = self.upload_policies(policies)
        self.create_index(policies)
        self.cleanup()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"  PDFs processed: {len(pdf_paths)}")
        print(f"  Policies extracted: {len(policies)}")
        print(f"  Total rules: {sum(p['rule_count'] for p in policies)}")
        print(f"  Policies uploaded: {uploaded}")
        print(f"  Time elapsed: {elapsed:.1f} seconds")
        print("="*60 + "\n")


def main():
    """Main entry point with .env support"""
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✓ Found .env file")
    else:
        print("⚠ No .env file found in current directory")
    
    # Get connection string
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    
    if not conn_str:
        print("\n" + "="*60)
        print("ERROR: Missing Azure Connection String")
        print("="*60)
        print("\nOption 1: Install python-dotenv and use .env file")
        print("  pip install python-dotenv")
        print("\nOption 2: Set environment variable directly")
        print('  export AZURE_STORAGE_CONNECTION_STRING="your_connection_string"')
        print("\nOption 3: Create .env file with:")
        print('  AZURE_STORAGE_CONNECTION_STRING=your_connection_string_here')
        print("="*60 + "\n")
        return 1
    
    print(f"✓ Connection string loaded (length: {len(conn_str)} chars)")
    
    # Run pipeline
    try:
        pipeline = SimpleAzurePipeline(conn_str)
        pipeline.run()
        return 0
    except Exception as e:
        print(f"\n✗ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())