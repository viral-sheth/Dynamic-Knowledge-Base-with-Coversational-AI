#!/usr/bin/env python3
"""
Intelligent PDF Filtering and Content Extraction System
Filters out irrelevant, duplicate, and low-quality PDFs before extraction

Key Features:
1. Pre-download filtering based on URL patterns
2. Content-based filtering after download
3. Duplicate detection and removal
4. Quality scoring and relevance assessment
5. Smart content extraction with deduplication

Author: Development Team
Date: October 2025
"""

import os
import re
import hashlib
import logging
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Set, Tuple, Optional
import requests
import fitz  # PyMuPDF
import json
from datetime import datetime
from collections import defaultdict
import difflib

class IntelligentPDFFilter:
    """Advanced PDF filtering system to ensure quality and relevance"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # URL-based filtering patterns
        self.high_value_url_patterns = [
            r'provider[_-]?manual',
            r'provider[_-]?guide',
            r'authorization',
            r'prior[_-]?auth',
            r'timely[_-]?filing',
            r'appeals?',
            r'claim[s]?[_-]?guide',
            r'billing[_-]?guide',
            r'coverage[_-]?guide',
            r'policy[_-]?manual',
            r'clinical[_-]?guidelines',
            r'medical[_-]?policy'
        ]
        
        self.low_value_url_patterns = [
            r'privacy[_-]?policy',
            r'terms[_-]?of[_-]?use',
            r'legal[_-]?notice',
            r'sample',
            r'test',
            r'demo',
            r'template',
            r'marketing',
            r'advertisement',
            r'brochure',
            r'newsletter',
            r'press[_-]?release',
            r'annual[_-]?report',
            r'financial[_-]?report'
        ]
        
        # Content-based filtering criteria
        self.essential_healthcare_terms = [
            'prior authorization', 'preauthorization', 'timely filing',
            'appeals process', 'claim submission', 'provider manual',
            'coverage determination', 'medical necessity', 'clinical criteria',
            'billing guidelines', 'reimbursement', 'CPT', 'ICD-10',
            'member benefits', 'covered services', 'exclusions'
        ]
        
        self.exclusion_content_patterns = [
            r'lorem ipsum',
            r'this is a test',
            r'sample document',
            r'coming soon',
            r'under construction',
            r'page intentionally left blank',
            r'for internal use only',
            r'confidential and proprietary',
            r'draft - not for distribution'
        ]
        
        # Quality thresholds
        self.min_content_length = 500  # Minimum meaningful content
        self.max_content_length = 1000000  # Maximum to avoid huge documents
        self.min_pages = 1
        self.max_pages = 300
        self.min_file_size = 2048  # 2KB
        self.max_file_size = 20 * 1024 * 1024  # 20MB
        
        # Results tracking
        self.filtered_urls = []
        self.rejected_urls = []
        self.content_hashes = {}
        self.extracted_content = {}
    
    def score_url_relevance(self, url: str) -> Tuple[int, str]:
        """
        Score URL relevance based on patterns
        
        Returns:
            (score, reason) where score > 0 is relevant
        """
        url_lower = url.lower()
        filename = os.path.basename(urlparse(url).path).lower()
        
        # Check for high-value patterns
        high_value_score = 0
        high_value_matches = []
        
        for pattern in self.high_value_url_patterns:
            if re.search(pattern, url_lower) or re.search(pattern, filename):
                high_value_score += 2
                high_value_matches.append(pattern)
        
        # Check for low-value patterns
        low_value_score = 0
        low_value_matches = []
        
        for pattern in self.low_value_url_patterns:
            if re.search(pattern, url_lower) or re.search(pattern, filename):
                low_value_score += 1
                low_value_matches.append(pattern)
        
        # Calculate final score
        final_score = high_value_score - low_value_score
        
        # Generate reason
        if high_value_matches:
            reason = f"High-value patterns: {', '.join(high_value_matches)}"
        elif low_value_matches:
            reason = f"Low-value patterns: {', '.join(low_value_matches)}"
        else:
            reason = "Neutral URL pattern"
        
        return final_score, reason
    
    def filter_urls_by_pattern(self, urls: List[str]) -> Tuple[List[str], List[str]]:
        """
        Filter URLs based on patterns before downloading
        
        Returns:
            (accepted_urls, rejected_urls)
        """
        accepted = []
        rejected = []
        
        for url in urls:
            score, reason = self.score_url_relevance(url)
            
            if score > 0:
                accepted.append(url)
                self.logger.info(f"✓ ACCEPTED: {url} | {reason}")
            else:
                rejected.append(url)
                self.logger.info(f"✗ REJECTED: {url} | {reason}")
        
        self.logger.info(f"URL filtering: {len(accepted)} accepted, {len(rejected)} rejected")
        return accepted, rejected
    
    def extract_clean_content(self, file_path: str, max_pages: int = 50) -> Dict:
        """
        Extract and clean content from PDF
        
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            doc = fitz.open(file_path)
            
            # Limit pages processed
            pages_to_process = min(max_pages, doc.page_count)
            
            # Extract text from all pages
            full_text = ""
            page_texts = []
            
            for page_num in range(pages_to_process):
                page = doc[page_num]
                page_text = page.get_text()
                
                # Basic cleaning
                page_text = re.sub(r'\n+', '\n', page_text)  # Remove excessive newlines
                page_text = re.sub(r'\s+', ' ', page_text)   # Normalize whitespace
                
                page_texts.append(page_text)
                full_text += page_text + "\n"
            
            doc.close()
            
            # Extract sections based on healthcare-specific patterns
            sections = self.extract_healthcare_sections(full_text)
            
            # Calculate content metrics
            content_length = len(full_text.strip())
            word_count = len(full_text.split())
            
            return {
                'full_text': full_text,
                'page_texts': page_texts,
                'sections': sections,
                'content_length': content_length,
                'word_count': word_count,
                'pages_processed': pages_to_process,
                'extraction_success': True
            }
            
        except Exception as e:
            self.logger.error(f"Content extraction failed for {file_path}: {e}")
            return {
                'extraction_success': False,
                'error': str(e)
            }
    
    def extract_healthcare_sections(self, text: str) -> Dict:
        """Extract specific healthcare-related sections from text"""
        sections = {}
        
        # Define section patterns
        section_patterns = {
            'prior_authorization': [
                r'prior authorization.*?(?=\n\s*[A-Z][^a-z]{10,}|\n\s*\d+\.|\Z)',
                r'preauthorization.*?(?=\n\s*[A-Z][^a-z]{10,}|\n\s*\d+\.|\Z)',
                r'authorization requirements.*?(?=\n\s*[A-Z][^a-z]{10,}|\n\s*\d+\.|\Z)'
            ],
            'timely_filing': [
                r'timely filing.*?(?=\n\s*[A-Z][^a-z]{10,}|\n\s*\d+\.|\Z)',
                r'claim submission deadline.*?(?=\n\s*[A-Z][^a-z]{10,}|\n\s*\d+\.|\Z)',
                r'filing requirements.*?(?=\n\s*[A-Z][^a-z]{10,}|\n\s*\d+\.|\Z)'
            ],
            'appeals': [
                r'appeals? process.*?(?=\n\s*[A-Z][^a-z]{10,}|\n\s*\d+\.|\Z)',
                r'dispute resolution.*?(?=\n\s*[A-Z][^a-z]{10,}|\n\s*\d+\.|\Z)',
                r'grievance.*?(?=\n\s*[A-Z][^a-z]{10,}|\n\s*\d+\.|\Z)'
            ],
            'claims_billing': [
                r'claim submission.*?(?=\n\s*[A-Z][^a-z]{10,}|\n\s*\d+\.|\Z)',
                r'billing guidelines.*?(?=\n\s*[A-Z][^a-z]{10,}|\n\s*\d+\.|\Z)',
                r'reimbursement.*?(?=\n\s*[A-Z][^a-z]{10,}|\n\s*\d+\.|\Z)'
            ]
        }
        
        text_lower = text.lower()
        
        for section_name, patterns in section_patterns.items():
            section_content = []
            
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    content = text[match.start():match.end()].strip()
                    if len(content) > 100:  # Only meaningful content
                        section_content.append(content)
            
            if section_content:
                sections[section_name] = section_content
        
        return sections
    
    def assess_content_quality(self, content: Dict) -> Dict:
        """
        Assess the quality and relevance of extracted content
        
        Returns:
            Quality assessment with scores and recommendations
        """
        if not content.get('extraction_success'):
            return {
                'quality_score': 0,
                'relevance_score': 0,
                'recommendation': 'reject',
                'reason': 'Extraction failed'
            }
        
        text = content['full_text'].lower()
        content_length = content['content_length']
        word_count = content['word_count']
        sections = content['sections']
        
        # Quality indicators
        quality_indicators = {
            'sufficient_length': content_length >= self.min_content_length,
            'reasonable_length': content_length <= self.max_content_length,
            'good_word_count': word_count >= 100,
            'has_sections': len(sections) > 0,
            'no_exclusion_patterns': True
        }
        
        # Check for exclusion patterns
        for pattern in self.exclusion_content_patterns:
            if re.search(pattern, text):
                quality_indicators['no_exclusion_patterns'] = False
                break
        
        # Relevance indicators
        relevance_score = 0
        found_terms = []
        
        for term in self.essential_healthcare_terms:
            if term.lower() in text:
                relevance_score += 1
                found_terms.append(term)
        
        # Calculate scores
        quality_score = sum(quality_indicators.values())
        
        # Make recommendation
        if quality_score >= 4 and relevance_score >= 2:
            recommendation = 'accept'
        elif quality_score >= 3 and relevance_score >= 1:
            recommendation = 'review'
        else:
            recommendation = 'reject'
        
        # Generate reason
        if recommendation == 'accept':
            reason = f"High quality content with {len(found_terms)} relevant terms"
        elif recommendation == 'review':
            reason = f"Moderate quality, needs review"
        else:
            reason = f"Low quality or irrelevant content"
        
        return {
            'quality_score': quality_score,
            'relevance_score': relevance_score,
            'recommendation': recommendation,
            'reason': reason,
            'quality_indicators': quality_indicators,
            'found_healthcare_terms': found_terms,
            'sections_found': list(sections.keys())
        }
    
    def detect_content_similarity(self, new_content: str, existing_contents: List[str], threshold: float = 0.8) -> Tuple[bool, float, int]:
        """
        Detect if new content is similar to existing content
        
        Returns:
            (is_duplicate, similarity_score, duplicate_index)
        """
        if not existing_contents:
            return False, 0.0, -1
        
        # Normalize content for comparison
        new_normalized = self.normalize_content(new_content)
        
        for i, existing in enumerate(existing_contents):
            existing_normalized = self.normalize_content(existing)
            
            # Calculate similarity using difflib
            similarity = difflib.SequenceMatcher(None, new_normalized, existing_normalized).ratio()
            
            if similarity >= threshold:
                return True, similarity, i
        
        return False, 0.0, -1
    
    def normalize_content(self, content: str) -> str:
        """Normalize content for similarity comparison"""
        # Convert to lowercase
        normalized = content.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove common variations
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation
        normalized = re.sub(r'\b(the|a|an|and|or|but|in|on|at|to|for|of|with|by)\b', '', normalized)  # Remove common words
        
        return normalized.strip()
    
    def process_pdf_batch_with_filtering(self, urls: List[str], max_pdfs: int = 20) -> Dict:
        """
        Process a batch of PDFs with comprehensive filtering
        
        Returns:
            Comprehensive results with filtering decisions
        """
        self.logger.info(f"Processing {len(urls)} URLs with intelligent filtering")
        
        # Step 1: URL-based filtering
        accepted_urls, rejected_urls = self.filter_urls_by_pattern(urls)
        
        # Limit to max_pdfs for processing
        processing_urls = accepted_urls[:max_pdfs]
        
        results = {
            'input_urls': len(urls),
            'url_filtered_accepted': len(accepted_urls),
            'url_filtered_rejected': len(rejected_urls),
            'processing_urls': len(processing_urls),
            'final_accepted': 0,
            'final_rejected': 0,
            'duplicates_removed': 0,
            'processed_pdfs': {},
            'accepted_content': {},
            'rejected_reasons': [],
            'processing_log': []
        }
        
        existing_contents = []
        
        for i, url in enumerate(processing_urls):
            self.logger.info(f"Processing PDF {i+1}/{len(processing_urls)}: {url}")
            
            try:
                # Download (simplified for this example)
                response = requests.get(url, timeout=30, stream=True)
                response.raise_for_status()
                
                # Save temporarily
                temp_path = f"temp_pdf_{i}.pdf"
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Extract content
                content = self.extract_clean_content(temp_path)
                
                if not content['extraction_success']:
                    results['final_rejected'] += 1
                    results['rejected_reasons'].append(f"Extraction failed: {content.get('error', 'Unknown')}")
                    os.remove(temp_path)
                    continue
                
                # Assess quality
                quality_assessment = self.assess_content_quality(content)
                
                # Check for duplicates
                is_duplicate, similarity, dup_index = self.detect_content_similarity(
                    content['full_text'], existing_contents
                )
                
                if is_duplicate:
                    results['duplicates_removed'] += 1
                    results['rejected_reasons'].append(f"Duplicate content (similarity: {similarity:.2f})")
                    os.remove(temp_path)
                    continue
                
                # Make final decision
                if quality_assessment['recommendation'] == 'accept':
                    results['final_accepted'] += 1
                    results['accepted_content'][url] = {
                        'content': content,
                        'quality_assessment': quality_assessment,
                        'file_size': os.path.getsize(temp_path)
                    }
                    existing_contents.append(content['full_text'])
                    self.logger.info(f"✓ ACCEPTED: {quality_assessment['reason']}")
                else:
                    results['final_rejected'] += 1
                    results['rejected_reasons'].append(quality_assessment['reason'])
                    self.logger.info(f"✗ REJECTED: {quality_assessment['reason']}")
                
                # Clean up
                os.remove(temp_path)
                
            except Exception as e:
                results['final_rejected'] += 1
                results['rejected_reasons'].append(f"Processing error: {str(e)[:100]}")
                self.logger.error(f"Error processing {url}: {e}")
        
        return results

def demonstrate_intelligent_filtering():
    """Demonstrate the intelligent filtering system"""
    
    print("🧠 Intelligent PDF Filtering Demonstration")
    print("=" * 60)
    
    # Sample URLs with mix of good and bad content
    sample_urls = [
        # High-value URLs
        "https://files.providernews.anthem.com/1661/2022-Provider-Manual-pages-44-113.pdf",
        "https://providers.anthem.com/docs/gpp/OH_CAID_ProviderManual.pdf",
        "https://providers.anthem.com/docs/gpp/NV_CAID_PriorAuthreq006648-22.pdf",
        
        # Medium-value URLs
        "https://www.uhc.com/content/dam/uhcdotcom/en/IndividualAndFamilies/PDF/2020-SBC-Sample.pdf",
        
        # Low-value URLs (should be filtered out)
        "https://www.uhc.com/content/dam/uhcdotcom/en/npp/OSPP-UHCPROVIDER-COM-EN.pdf",  # Privacy policy
        "https://www.uhc.com/content/dam/uhcdotcom/en/npp/TOU-UHCPROVIDER-COM-EN.pdf",   # Terms of use
        
        # Test URLs
        "https://example.com/test-document.pdf",
        "https://provider.com/marketing-brochure.pdf",
        "https://healthplan.com/annual-report.pdf"
    ]
    
    filter_system = IntelligentPDFFilter()
    
    # Test URL filtering
    print("\n🔍 URL Pattern Filtering:")
    accepted, rejected = filter_system.filter_urls_by_pattern(sample_urls)
    
    print(f"\nURL Filtering Results:")
    print(f"  Input URLs: {len(sample_urls)}")
    print(f"  Accepted: {len(accepted)}")
    print(f"  Rejected: {len(rejected)}")
    
    # Test content processing (only on accepted URLs that are reachable)
    reachable_urls = [
        "https://files.providernews.anthem.com/1661/2022-Provider-Manual-pages-44-113.pdf",
        "https://providers.anthem.com/docs/gpp/OH_CAID_ProviderManual.pdf",
        "https://www.uhc.com/content/dam/uhcdotcom/en/IndividualAndFamilies/PDF/2020-SBC-Sample.pdf"
    ]
    
    print(f"\n🔬 Content Quality Analysis:")
    print(f"Testing {len(reachable_urls)} reachable URLs...")
    
    results = filter_system.process_pdf_batch_with_filtering(reachable_urls, max_pdfs=5)
    
    print(f"\n📊 COMPREHENSIVE FILTERING RESULTS:")
    print(f"Input URLs: {results['input_urls']}")
    print(f"URL-level accepted: {results['url_filtered_accepted']}")
    print(f"URL-level rejected: {results['url_filtered_rejected']}")
    print(f"Content-level accepted: {results['final_accepted']}")
    print(f"Content-level rejected: {results['final_rejected']}")
    print(f"Duplicates removed: {results['duplicates_removed']}")
    
    if results['rejected_reasons']:
        print(f"\n❌ Rejection Reasons:")
        for reason in set(results['rejected_reasons']):
            count = results['rejected_reasons'].count(reason)
            print(f"  - {reason} ({count}x)")
    
    # Show accepted content summary
    if results['accepted_content']:
        print(f"\n✅ ACCEPTED CONTENT SUMMARY:")
        for url, data in results['accepted_content'].items():
            assessment = data['quality_assessment']
            content = data['content']
            print(f"\nURL: {url}")
            print(f"  Quality Score: {assessment['quality_score']}/5")
            print(f"  Relevance Score: {assessment['relevance_score']}")
            print(f"  Healthcare Terms: {len(assessment['found_healthcare_terms'])}")
            print(f"  Sections Found: {assessment['sections_found']}")
            print(f"  Content Length: {content['content_length']:,} chars")
            print(f"  Word Count: {content['word_count']:,}")
    
    return results

if __name__ == "__main__":
    demonstrate_intelligent_filtering()