#!/usr/bin/env python3
"""
Regional Coverage Analyzer
Analyzes geographic coverage patterns in healthcare payer PDF discoveries

This script:
1. Analyzes existing PDFs for regional/state-specific content
2. Maps payer coverage areas and regional variations
3. Identifies gaps in regional coverage
4. Provides strategies for comprehensive regional discovery

Author: Development Team
Date: October 2025
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd
import json
from collections import defaultdict, Counter
from urllib.parse import urlparse

class RegionalCoverageAnalyzer:
    """Analyzes regional coverage patterns in healthcare payer discoveries"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # US States and territories
        self.us_states = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
            'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
            'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
            'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
            'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
            'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
            'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
            'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
            'DC': 'District of Columbia', 'PR': 'Puerto Rico', 'VI': 'Virgin Islands', 'GU': 'Guam'
        }
        
        # Major healthcare payers and their known coverage areas
        self.payer_coverage_map = {
            'United Healthcare': {
                'type': 'National',
                'strong_regions': ['All US States'],
                'market_segments': ['Commercial', 'Medicare', 'Medicaid'],
                'regional_variations': True
            },
            'Anthem/Elevance Health': {
                'type': 'Multi-State',
                'strong_regions': ['CA', 'CO', 'CT', 'GA', 'IN', 'KY', 'ME', 'MO', 'NV', 'NH', 'NY', 'OH', 'VA', 'WI'],
                'market_segments': ['Commercial', 'Medicare', 'Medicaid'],
                'regional_variations': True
            },
            'Aetna': {
                'type': 'National',
                'strong_regions': ['All US States'],
                'market_segments': ['Commercial', 'Medicare'],
                'regional_variations': True
            },
            'Kaiser Permanente': {
                'type': 'Regional',
                'strong_regions': ['CA', 'CO', 'GA', 'HI', 'MD', 'OR', 'VA', 'WA', 'DC'],
                'market_segments': ['Commercial', 'Medicare'],
                'regional_variations': True
            },
            'Centene Corporation': {
                'type': 'Multi-State',
                'strong_regions': ['AZ', 'AR', 'CA', 'FL', 'GA', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'MI', 'MS', 'MO', 'NV', 'NH', 'NM', 'NY', 'NC', 'OH', 'OK', 'PA', 'SC', 'TN', 'TX', 'WA', 'WI'],
                'market_segments': ['Medicaid', 'Medicare', 'Health Insurance Marketplace'],
                'regional_variations': True
            }
        }
        
        # Regional URL patterns
        self.regional_url_patterns = [
            r'/([A-Z]{2})[_-]',  # State codes in URLs
            r'([A-Z]{2})[_-]CAID',  # Medicaid state patterns
            r'/([a-z]{2})/',  # Lowercase state codes
            r'([A-Z]{2})\.pdf',  # State codes in filenames
            r'state[_-]([a-z]{2})',  # State-specific patterns
            r'region[_-]([a-z]+)'  # Regional patterns
        ]
        
        # Content patterns for regional identification
        self.regional_content_patterns = [
            r'(?i)(\w+)\s+state\s+specific',
            r'(?i)applicable\s+in\s+(\w+)',
            r'(?i)(\w+)\s+medicaid',
            r'(?i)(\w+)\s+department\s+of\s+health',
            r'(?i)(\w+)\s+insurance\s+commission',
            r'(?i)state\s+of\s+(\w+)'
        ]
    
    def extract_regions_from_url(self, url: str) -> Set[str]:
        """Extract regional indicators from URL"""
        regions = set()
        url_upper = url.upper()
        
        # Check for state codes in URL patterns
        for pattern in self.regional_url_patterns:
            matches = re.findall(pattern, url)
            for match in matches:
                state_code = match.upper()
                if state_code in self.us_states:
                    regions.add(state_code)
        
        # Check for full state names in URL
        for state_code, state_name in self.us_states.items():
            if state_name.upper().replace(' ', '').replace('-', '') in url_upper.replace('-', '').replace('_', ''):
                regions.add(state_code)
        
        return regions
    
    def extract_regions_from_content(self, text: str) -> Set[str]:
        """Extract regional indicators from PDF content"""
        regions = set()
        text_lower = text.lower()
        
        # Direct state code mentions
        for state_code, state_name in self.us_states.items():
            # Check for state codes
            if re.search(rf'\b{state_code.lower()}\b', text_lower):
                regions.add(state_code)
            
            # Check for full state names
            if state_name.lower() in text_lower:
                regions.add(state_code)
        
        # Pattern-based extraction
        for pattern in self.regional_content_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Try to match extracted text to states
                match_clean = match.strip().upper()
                if match_clean in self.us_states:
                    regions.add(match_clean)
                else:
                    # Try to find state by name
                    for state_code, state_name in self.us_states.items():
                        if match_clean.lower() == state_name.lower():
                            regions.add(state_code)
        
        return regions
    
    def analyze_pdf_regional_coverage(self, pdf_data: Dict) -> Dict:
        """Analyze regional coverage of a single PDF"""
        url = pdf_data.get('url', '')
        content = pdf_data.get('content', {})
        text = content.get('full_text', '')
        
        # Extract regions from URL
        url_regions = self.extract_regions_from_url(url)
        
        # Extract regions from content
        content_regions = self.extract_regions_from_content(text)
        
        # Combine and analyze
        all_regions = url_regions.union(content_regions)
        
        # Determine scope
        if len(all_regions) == 0:
            scope = 'National/Unspecified'
        elif len(all_regions) == 1:
            scope = 'State-Specific'
        elif len(all_regions) <= 5:
            scope = 'Multi-State'
        else:
            scope = 'National'
        
        return {
            'url_regions': list(url_regions),
            'content_regions': list(content_regions),
            'all_regions': list(all_regions),
            'region_count': len(all_regions),
            'scope': scope,
            'coverage_type': self.determine_coverage_type(url, text)
        }
    
    def determine_coverage_type(self, url: str, text: str) -> str:
        """Determine the type of regional coverage"""
        url_lower = url.lower()
        text_lower = text.lower()
        
        # Check for specific coverage types
        if 'medicaid' in url_lower or 'medicaid' in text_lower:
            return 'Medicaid'
        elif 'medicare' in url_lower or 'medicare' in text_lower:
            return 'Medicare'
        elif 'commercial' in url_lower or 'commercial' in text_lower:
            return 'Commercial'
        elif 'marketplace' in url_lower or 'marketplace' in text_lower:
            return 'Marketplace'
        else:
            return 'General'
    
    def analyze_payer_regional_coverage(self, payer_name: str, pdf_list: List[Dict]) -> Dict:
        """Analyze regional coverage for a specific payer"""
        regional_analysis = {
            'payer_name': payer_name,
            'total_pdfs': len(pdf_list),
            'regional_breakdown': {},
            'coverage_gaps': [],
            'scope_distribution': Counter(),
            'coverage_type_distribution': Counter(),
            'regional_completeness_score': 0.0
        }
        
        # Expected coverage based on payer profile
        expected_regions = set()
        if payer_name in self.payer_coverage_map:
            payer_info = self.payer_coverage_map[payer_name]
            if payer_info['strong_regions'] == ['All US States']:
                expected_regions = set(self.us_states.keys())
            else:
                expected_regions = set(payer_info['strong_regions'])
        
        # Analyze each PDF
        found_regions = set()
        region_pdf_count = defaultdict(int)
        
        for pdf_data in pdf_list:
            analysis = self.analyze_pdf_regional_coverage(pdf_data)
            
            # Track scope distribution
            regional_analysis['scope_distribution'][analysis['scope']] += 1
            regional_analysis['coverage_type_distribution'][analysis['coverage_type']] += 1
            
            # Track regional coverage
            for region in analysis['all_regions']:
                found_regions.add(region)
                region_pdf_count[region] += 1
        
        # Calculate regional breakdown
        regional_analysis['regional_breakdown'] = dict(region_pdf_count)
        regional_analysis['covered_regions'] = list(found_regions)
        regional_analysis['covered_region_count'] = len(found_regions)
        
        # Identify coverage gaps
        if expected_regions:
            missing_regions = expected_regions - found_regions
            regional_analysis['coverage_gaps'] = list(missing_regions)
            regional_analysis['expected_regions'] = list(expected_regions)
            
            # Calculate completeness score
            if expected_regions:
                regional_analysis['regional_completeness_score'] = len(found_regions) / len(expected_regions)
        
        return regional_analysis
    
    def generate_regional_discovery_strategy(self, payer_analysis: Dict) -> Dict:
        """Generate strategy to improve regional coverage"""
        payer_name = payer_analysis['payer_name']
        gaps = payer_analysis.get('coverage_gaps', [])
        
        strategy = {
            'payer_name': payer_name,
            'current_coverage': payer_analysis['regional_completeness_score'],
            'missing_regions': gaps,
            'recommended_actions': [],
            'targeted_urls': [],
            'search_patterns': []
        }
        
        if gaps:
            # Generate specific search strategies for missing regions
            for region in gaps:
                state_name = self.us_states[region]
                
                # URL patterns to try
                url_patterns = [
                    f"/{region}_",
                    f"/{region.lower()}/",
                    f"/{state_name.lower().replace(' ', '-')}/",
                    f"/{region}_CAID",
                    f"/state-{region.lower()}/",
                    f"/regional/{region.lower()}/"
                ]
                
                # Search terms
                search_terms = [
                    f"{state_name} provider manual",
                    f"{region} medicaid",
                    f"{state_name} specific guidelines",
                    f"{region} authorization requirements"
                ]
                
                strategy['search_patterns'].extend(url_patterns)
                strategy['recommended_actions'].append(
                    f"Search for {state_name} ({region}) specific content"
                )
        
        # General recommendations
        if payer_analysis['regional_completeness_score'] < 0.5:
            strategy['recommended_actions'].append("Increase BFS depth for regional discovery")
            strategy['recommended_actions'].append("Check state-specific subdomains")
            strategy['recommended_actions'].append("Search provider portal state selection pages")
        
        return strategy

def analyze_sample_regional_coverage():
    """Analyze regional coverage using our sample data"""
    
    print("🗺️  REGIONAL COVERAGE ANALYSIS")
    print("=" * 60)
    
    # Sample PDFs from our previous tests with regional indicators
    sample_pdfs = [
        {
            'url': 'https://providers.anthem.com/docs/gpp/OH_CAID_ProviderManual.pdf',
            'content': {'full_text': 'Ohio Medicaid provider manual with state-specific guidelines for Ohio healthcare providers'}
        },
        {
            'url': 'https://providers.anthem.com/docs/gpp/CA_CAID_ProviderManual.pdf', 
            'content': {'full_text': 'California Medicaid CAID provider manual for California state requirements'}
        },
        {
            'url': 'https://providers.anthem.com/docs/gpp/NY_ABC_CAID_ProviderManual.pdf',
            'content': {'full_text': 'New York Medicaid provider guidelines and New York state procedures'}
        },
        {
            'url': 'https://providers.anthem.com/docs/gpp/VA_CAID_ProviderManual.pdf',
            'content': {'full_text': 'Virginia Medicaid requirements for healthcare providers in Virginia'}
        },
        {
            'url': 'https://providers.anthem.com/docs/gpp/WI_CAID_Provider_Manual.pdf',
            'content': {'full_text': 'Wisconsin Medicaid provider manual with Wisconsin-specific authorization requirements'}
        },
        {
            'url': 'https://providers.anthem.com/docs/gpp/NV_CAID_PriorAuthreq006648-22.pdf',
            'content': {'full_text': 'Nevada prior authorization requirements for Nevada Medicaid providers'}
        },
        {
            'url': 'https://www.uhc.com/content/dam/uhcdotcom/en/IndividualAndFamilies/PDF/2020-SBC-Sample.pdf',
            'content': {'full_text': 'National summary of benefits and coverage applicable across all states'}
        },
        {
            'url': 'https://providers.kp.org/content/dam/kp-providers/en/co/pdf/doula-coverage-notification.pdf',
            'content': {'full_text': 'Colorado doula coverage notification for Colorado Kaiser Permanente members'}
        }
    ]
    
    analyzer = RegionalCoverageAnalyzer()
    
    # Group by payer
    payer_pdfs = {
        'Anthem/Elevance Health': [pdf for pdf in sample_pdfs if 'anthem.com' in pdf['url']],
        'United Healthcare': [pdf for pdf in sample_pdfs if 'uhc.com' in pdf['url']],
        'Kaiser Permanente': [pdf for pdf in sample_pdfs if 'kp.org' in pdf['url']]
    }
    
    print(f"Analyzing {len(sample_pdfs)} sample PDFs across {len(payer_pdfs)} payers\n")
    
    # Analyze each payer
    for payer_name, pdfs in payer_pdfs.items():
        print(f"🏢 {payer_name}")
        print("-" * 40)
        
        analysis = analyzer.analyze_payer_regional_coverage(payer_name, pdfs)
        
        print(f"Total PDFs: {analysis['total_pdfs']}")
        print(f"Regions covered: {analysis['covered_region_count']}")
        print(f"Regional completeness: {analysis['regional_completeness_score']:.1%}")
        
        if analysis['covered_regions']:
            print(f"Covered regions: {', '.join(analysis['covered_regions'])}")
        
        if analysis['coverage_gaps']:
            print(f"Missing regions: {', '.join(analysis['coverage_gaps'][:5])}{'...' if len(analysis['coverage_gaps']) > 5 else ''}")
        
        # Scope distribution
        print(f"Scope distribution:")
        for scope, count in analysis['scope_distribution'].items():
            print(f"  {scope}: {count}")
        
        # Generate improvement strategy
        strategy = analyzer.generate_regional_discovery_strategy(analysis)
        
        if strategy['recommended_actions']:
            print(f"Recommendations:")
            for action in strategy['recommended_actions'][:3]:
                print(f"  • {action}")
        
        print()
    
    # Overall analysis
    all_regions = set()
    total_pdfs = 0
    
    for payer_name, pdfs in payer_pdfs.items():
        analysis = analyzer.analyze_payer_regional_coverage(payer_name, pdfs)
        all_regions.update(analysis['covered_regions'])
        total_pdfs += len(pdfs)
    
    print(f"📊 OVERALL REGIONAL COVERAGE")
    print("-" * 40)
    print(f"Total PDFs analyzed: {total_pdfs}")
    print(f"Unique regions covered: {len(all_regions)}")
    print(f"Overall US coverage: {len(all_regions)}/50 states ({len(all_regions)/50:.1%})")
    print(f"Covered regions: {', '.join(sorted(all_regions))}")
    
    # Identify major gaps
    missing_regions = set(analyzer.us_states.keys()) - all_regions
    if missing_regions:
        print(f"Major gaps: {', '.join(sorted(list(missing_regions)[:10]))}{'...' if len(missing_regions) > 10 else ''}")
    
    # Regional coverage assessment
    print(f"\n🎯 REGIONAL COVERAGE ASSESSMENT")
    print("-" * 40)
    
    coverage_score = len(all_regions) / 50
    
    if coverage_score >= 0.8:
        grade = "Excellent"
    elif coverage_score >= 0.6:
        grade = "Good"
    elif coverage_score >= 0.4:
        grade = "Fair"
    else:
        grade = "Poor"
    
    print(f"Coverage Grade: {grade} ({coverage_score:.1%})")
    
    # Recommendations for comprehensive coverage
    print(f"\n🚀 COMPREHENSIVE COVERAGE STRATEGY")
    print("-" * 40)
    print("To achieve complete regional coverage:")
    print("1. Target state-specific provider portals")
    print("2. Search for Medicaid-specific documentation")
    print("3. Check regional health plan variations")
    print("4. Explore state insurance commission resources")
    print("5. Use geographic crawling patterns in BFS")
    
    return {
        'total_regions_covered': len(all_regions),
        'coverage_percentage': coverage_score,
        'missing_regions': list(missing_regions),
        'grade': grade
    }

if __name__ == "__main__":
    analyze_sample_regional_coverage()