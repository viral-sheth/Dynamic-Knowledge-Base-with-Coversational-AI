#!/usr/bin/env python3
"""
PDF Quality Analyzer and Content Filter
Analyzes discovered PDFs for relevance, quality, and content value

This script:
1. Downloads and analyzes PDFs to check validity
2. Extracts metadata and content previews
3. Filters out redundant, dummy, or low-value content
4. Categorizes PDFs by relevance and usefulness
5. Provides content deduplication

Author: Development Team
Date: October 2025
"""

import os
import time
import hashlib
import logging
import requests
import fitz  # PyMuPDF
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
import pandas as pd
import re
from collections import Counter
from typing import Dict, List, Set, Tuple
import json

class PDFQualityAnalyzer:
    """Analyzes PDF quality and filters out irrelevant content"""
    
    def __init__(self, download_dir="pdf_analysis"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Content categories for filtering
        self.relevant_categories = {
            'prior_authorization': [
                'prior authorization', 'preauthorization', 'pre-auth', 'authorization criteria',
                'approval requirements', 'auth guidelines', 'coverage determination'
            ],
            'timely_filing': [
                'timely filing', 'claim submission deadline', 'filing requirement',
                'submission timeline', 'deadline', 'time limit', 'filing window'
            ],
            'appeals': [
                'appeals process', 'claim appeal', 'dispute resolution', 'appeal procedure',
                'grievance', 'complaint', 'review process', 'reconsideration'
            ],
            'provider_manual': [
                'provider manual', 'provider guide', 'provider handbook', 'provider reference',
                'provider policies', 'provider procedures', 'clinical guidelines'
            ],
            'claims_billing': [
                'claims', 'billing', 'reimbursement', 'payment', 'coding', 'claim form',
                'billing guidelines', 'claim submission', 'claim processing'
            ],
            'coverage_benefits': [
                'coverage', 'benefits', 'covered services', 'benefit summary',
                'plan benefits', 'coverage guidelines', 'covered procedures'
            ],
            'forms_documents': [
                'form', 'application', 'request form', 'enrollment', 'registration',
                'member form', 'provider form', 'claim form'
            ]
        }
        
        # Patterns that indicate low-value or dummy content
        self.exclusion_patterns = [
            'test', 'sample', 'demo', 'template', 'example',
            'lorem ipsum', 'placeholder', 'draft', 'temporary',
            'coming soon', 'under construction', 'not available',
            'privacy policy', 'terms of use', 'legal notice',
            'marketing', 'advertisement', 'promotional'
        ]
        
        # File size limits (in bytes)
        self.min_file_size = 1024  # 1KB minimum
        self.max_file_size = 50 * 1024 * 1024  # 50MB maximum
        
        # Results tracking
        self.analysis_results = {}
        self.content_hashes = {}
        self.duplicate_groups = []
    
    def download_pdf(self, url: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """
        Download PDF and return success status, file path, and error message
        
        Returns:
            (success, file_path, error_message)
        """
        try:
            # Create filename from URL
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            # Sanitize filename
            filename = re.sub(r'[^\w\-_\.]', '_', filename)
            file_path = self.download_dir / filename
            
            # Skip if already downloaded
            if file_path.exists():
                return True, str(file_path), ""
            
            # Download with headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
                return False, "", f"Not a PDF file (content-type: {content_type})"
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length:
                size = int(content_length)
                if size < self.min_file_size:
                    return False, "", f"File too small ({size} bytes)"
                if size > self.max_file_size:
                    return False, "", f"File too large ({size} bytes)"
            
            # Save file
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Final size check
            actual_size = file_path.stat().st_size
            if actual_size < self.min_file_size:
                file_path.unlink()
                return False, "", f"Downloaded file too small ({actual_size} bytes)"
            
            return True, str(file_path), ""
            
        except Exception as e:
            return False, "", str(e)
    
    def extract_pdf_metadata(self, file_path: str) -> Dict:
        """Extract metadata and basic info from PDF"""
        try:
            doc = fitz.open(file_path)
            
            # Basic metadata
            metadata = {
                'page_count': doc.page_count,
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', ''),
                'file_size': os.path.getsize(file_path),
                'encrypted': doc.needs_pass
            }
            
            # Extract first few pages of text for analysis
            text_sample = ""
            max_pages = min(3, doc.page_count)
            
            for page_num in range(max_pages):
                page = doc[page_num]
                text_sample += page.get_text()
                if len(text_sample) > 2000:  # Limit sample size
                    break
            
            metadata['text_sample'] = text_sample[:2000]
            metadata['estimated_text_length'] = len(text_sample) * (doc.page_count / max_pages)
            
            doc.close()
            return metadata
            
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_content_hash(self, text: str) -> str:
        """Calculate hash of normalized content for duplicate detection"""
        # Normalize text for comparison
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def categorize_content(self, metadata: Dict) -> Dict:
        """Categorize PDF content and assess relevance"""
        text = metadata.get('text_sample', '').lower()
        title = metadata.get('title', '').lower()
        
        # Combine text sources for analysis
        combined_text = f"{title} {text}"
        
        # Check relevance categories
        category_scores = {}
        for category, keywords in self.relevant_categories.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                category_scores[category] = score
        
        # Check for exclusion patterns
        exclusion_score = sum(1 for pattern in self.exclusion_patterns if pattern in combined_text)
        
        # Determine primary category
        primary_category = max(category_scores.keys(), key=category_scores.get) if category_scores else 'uncategorized'
        
        # Calculate relevance score
        relevance_score = sum(category_scores.values()) - (exclusion_score * 2)
        
        # Quality indicators
        has_meaningful_title = len(metadata.get('title', '').strip()) > 10
        sufficient_content = metadata.get('estimated_text_length', 0) > 1000
        reasonable_pages = 1 <= metadata.get('page_count', 0) <= 500
        
        quality_score = sum([
            has_meaningful_title,
            sufficient_content,
            reasonable_pages,
            exclusion_score == 0  # No exclusion patterns
        ])
        
        return {
            'primary_category': primary_category,
            'category_scores': category_scores,
            'exclusion_score': exclusion_score,
            'relevance_score': relevance_score,
            'quality_score': quality_score,
            'is_relevant': relevance_score > 0 and quality_score >= 2,
            'quality_indicators': {
                'has_meaningful_title': has_meaningful_title,
                'sufficient_content': sufficient_content,
                'reasonable_pages': reasonable_pages,
                'no_exclusions': exclusion_score == 0
            }
        }
    
    def detect_duplicates(self) -> List[List[str]]:
        """Detect duplicate content across PDFs"""
        # Group by content hash
        hash_groups = {}
        for url, analysis in self.analysis_results.items():
            if 'content_hash' in analysis:
                content_hash = analysis['content_hash']
                if content_hash not in hash_groups:
                    hash_groups[content_hash] = []
                hash_groups[content_hash].append(url)
        
        # Find groups with duplicates
        duplicate_groups = [urls for urls in hash_groups.values() if len(urls) > 1]
        
        self.duplicate_groups = duplicate_groups
        return duplicate_groups
    
    def analyze_pdf_batch(self, pdf_urls: List[str], max_downloads: int = 20) -> Dict:
        """Analyze a batch of PDF URLs"""
        self.logger.info(f"Analyzing {len(pdf_urls)} PDF URLs (max downloads: {max_downloads})")
        
        results = {
            'total_urls': len(pdf_urls),
            'analyzed_count': 0,
            'download_success': 0,
            'download_failed': 0,
            'relevant_pdfs': 0,
            'irrelevant_pdfs': 0,
            'duplicate_groups': 0,
            'analysis_results': {}
        }
        
        downloaded_count = 0
        
        for i, url in enumerate(pdf_urls):
            if downloaded_count >= max_downloads:
                self.logger.info(f"Reached download limit ({max_downloads})")
                break
                
            self.logger.info(f"Analyzing PDF {i+1}/{len(pdf_urls)}: {url}")
            
            # Download PDF
            success, file_path, error = self.download_pdf(url)
            
            if not success:
                self.logger.warning(f"Download failed: {error}")
                results['analysis_results'][url] = {
                    'download_success': False,
                    'error': error
                }
                results['download_failed'] += 1
                continue
            
            downloaded_count += 1
            results['download_success'] += 1
            
            # Extract metadata
            metadata = self.extract_pdf_metadata(file_path)
            
            if 'error' in metadata:
                self.logger.warning(f"Metadata extraction failed: {metadata['error']}")
                results['analysis_results'][url] = {
                    'download_success': True,
                    'file_path': file_path,
                    'metadata_error': metadata['error']
                }
                continue
            
            # Categorize content
            categorization = self.categorize_content(metadata)
            
            # Calculate content hash
            content_hash = self.calculate_content_hash(metadata.get('text_sample', ''))
            
            # Store complete analysis
            analysis = {
                'download_success': True,
                'file_path': file_path,
                'metadata': metadata,
                'categorization': categorization,
                'content_hash': content_hash,
                'url': url
            }
            
            self.analysis_results[url] = analysis
            results['analysis_results'][url] = analysis
            results['analyzed_count'] += 1
            
            # Track relevance
            if categorization['is_relevant']:
                results['relevant_pdfs'] += 1
            else:
                results['irrelevant_pdfs'] += 1
            
            # Brief progress update
            category = categorization['primary_category']
            relevance = "✓" if categorization['is_relevant'] else "✗"
            pages = metadata.get('page_count', 0)
            self.logger.info(f"  {relevance} {category} | {pages} pages | {metadata.get('file_size', 0)/1024:.1f}KB")
        
        # Detect duplicates
        duplicate_groups = self.detect_duplicates()
        results['duplicate_groups'] = len(duplicate_groups)
        results['duplicates_detail'] = duplicate_groups
        
        return results
    
    def generate_quality_report(self, results: Dict) -> str:
        """Generate a comprehensive quality report"""
        report = []
        report.append("🔍 PDF QUALITY ANALYSIS REPORT")
        report.append("=" * 50)
        
        # Summary statistics
        total = results['total_urls']
        analyzed = results['analyzed_count']
        relevant = results['relevant_pdfs']
        irrelevant = results['irrelevant_pdfs']
        duplicates = results['duplicate_groups']
        
        report.append(f"\n📊 SUMMARY STATISTICS")
        report.append(f"Total PDF URLs: {total}")
        report.append(f"Successfully downloaded: {results['download_success']}")
        report.append(f"Download failures: {results['download_failed']}")
        report.append(f"Successfully analyzed: {analyzed}")
        report.append(f"Relevant PDFs: {relevant} ({relevant/analyzed*100:.1f}%)" if analyzed > 0 else "Relevant PDFs: 0")
        report.append(f"Irrelevant PDFs: {irrelevant} ({irrelevant/analyzed*100:.1f}%)" if analyzed > 0 else "Irrelevant PDFs: 0")
        report.append(f"Duplicate groups: {duplicates}")
        
        # Category breakdown
        if analyzed > 0:
            category_counts = Counter()
            for analysis in results['analysis_results'].values():
                if 'categorization' in analysis:
                    category = analysis['categorization']['primary_category']
                    category_counts[category] += 1
            
            report.append(f"\n📂 CONTENT CATEGORIES")
            for category, count in category_counts.most_common():
                percentage = count / analyzed * 100
                report.append(f"{category}: {count} ({percentage:.1f}%)")
        
        # Quality indicators
        if analyzed > 0:
            quality_metrics = {
                'has_meaningful_title': 0,
                'sufficient_content': 0,
                'reasonable_pages': 0,
                'no_exclusions': 0
            }
            
            for analysis in results['analysis_results'].values():
                if 'categorization' in analysis:
                    indicators = analysis['categorization']['quality_indicators']
                    for metric, value in indicators.items():
                        if value:
                            quality_metrics[metric] += 1
            
            report.append(f"\n✅ QUALITY METRICS")
            for metric, count in quality_metrics.items():
                percentage = count / analyzed * 100
                report.append(f"{metric}: {count}/{analyzed} ({percentage:.1f}%)")
        
        # Top relevant PDFs
        relevant_pdfs = [
            (url, analysis) for url, analysis in results['analysis_results'].items()
            if 'categorization' in analysis and analysis['categorization']['is_relevant']
        ]
        
        if relevant_pdfs:
            # Sort by relevance score
            relevant_pdfs.sort(key=lambda x: x[1]['categorization']['relevance_score'], reverse=True)
            
            report.append(f"\n🏆 TOP RELEVANT PDFs")
            for i, (url, analysis) in enumerate(relevant_pdfs[:5], 1):
                category = analysis['categorization']['primary_category']
                score = analysis['categorization']['relevance_score']
                pages = analysis['metadata'].get('page_count', 0)
                title = analysis['metadata'].get('title', 'No title')[:50]
                report.append(f"{i}. {category} (score: {score}) | {pages}p | {title}")
                report.append(f"   URL: {url}")
        
        # Duplicate information
        if duplicates > 0:
            report.append(f"\n🔄 DUPLICATE CONTENT")
            for i, group in enumerate(results['duplicates_detail'][:3], 1):
                report.append(f"Duplicate group {i}: {len(group)} PDFs")
                for url in group[:2]:  # Show first 2 URLs
                    report.append(f"  - {url}")
                if len(group) > 2:
                    report.append(f"  ... and {len(group) - 2} more")
        
        return "\n".join(report)

def test_pdf_quality_analysis():
    """Test PDF quality analysis with discovered URLs"""
    
    # Sample PDFs from our previous discoveries
    sample_pdfs = [
        # United Healthcare PDFs
        "https://www.uhc.com/content/dam/uhcdotcom/en/npp/OSPP-UHCPROVIDER-COM-EN.pdf",
        "https://www.uhc.com/content/dam/uhcdotcom/en/npp/TOU-UHCPROVIDER-COM-EN.pdf",
        "https://www.uhc.com/content/dam/uhcdotcom/en/IndividualAndFamilies/PDF/2020-SBC-Sample.pdf",
        
        # Kaiser Permanente PDFs  
        "https://providers.kp.org/content/dam/kp-providers/en/co/pdf/doula-coverage-notification.pdf",
        "https://providers.kp.org/content/dam/kp-providers/en/co/pdf/behavioral-health-admissions-process.pdf",
        "https://providers.kp.org/content/dam/kp-providers/en/co/pdf/pain-diagnoses-en.pdf",
        
        # Anthem PDFs (from basic crawler test)
        "https://files.providernews.anthem.com/1661/2022-Provider-Manual-pages-44-113.pdf",
        "https://providers.anthem.com/docs/gpp/OH_CAID_ProviderManual.pdf",
        "https://providers.anthem.com/docs/gpp/NV_CAID_PriorAuthreq006648-22.pdf"
    ]
    
    print("🔬 Testing PDF Quality Analysis")
    print("=" * 50)
    print(f"Analyzing {len(sample_pdfs)} sample PDFs...")
    
    analyzer = PDFQualityAnalyzer()
    results = analyzer.analyze_pdf_batch(sample_pdfs, max_downloads=10)
    
    # Generate and display report
    report = analyzer.generate_quality_report(results)
    print(f"\n{report}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"pdf_quality_analysis_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    test_pdf_quality_analysis()