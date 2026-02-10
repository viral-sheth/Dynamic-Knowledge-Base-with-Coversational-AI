"""
High-Quality Healthcare Policy PDF Processor
Extracts structured, clean data from healthcare payer PDFs
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
import pdfplumber
from datetime import datetime

class HealthcarePolicyExtractor:
    """Extract structured healthcare policies from PDFs"""
    
    def __init__(self):
        # Healthcare-specific patterns for rule extraction
        self.patterns = {
            'prior_authorization': [
                r'(?:prior authorization|pre-?authorization|preauth)(?:\s+(?:is\s+)?required\s+for|:)\s*([^.]+?)(?:\.|$)',
                r'requires?\s+(?:prior\s+)?authorization:?\s*([^.]+?)(?:\.|$)',
                r'must\s+(?:be\s+)?(?:pre-?authorized|obtain\s+authorization)(?:\s+for)?\s*([^.]+?)(?:\.|$)',
                r'authorization\s+required\s+for:?\s*([^.]+?)(?:\.|$)',
            ],
            'timely_filing': [
                r'(?:submit|file)\s+(?:claims?|bills?)\s+within\s+(\d+)\s+(days?|months?|years?)',
                r'(?:timely\s+)?filing\s+(?:deadline|limit|requirement):?\s*(\d+)\s+(days?|months?)',
                r'claims?\s+must\s+be\s+(?:submitted|filed)\s+within\s+(\d+)\s+(days?|months?)',
                r'(\d+)[- ]day\s+filing\s+(?:deadline|limit|requirement)',
            ],
            'appeals': [
                r'(?:appeal|dispute|reconsideration)(?:\s+must\s+be\s+(?:filed|submitted))?\s+within\s+(\d+)\s+(days?|months?)',
                r'(?:appeal|dispute)\s+(?:process|procedure|deadline):?\s*([^.]+?)(?:\.|$)',
                r'to\s+(?:appeal|dispute)\s+(?:a\s+)?(?:claim|decision)[,:]?\s*([^.]+?)(?:\.|$)',
            ],
            'claims_submission': [
                r'claims?\s+(?:should|must)\s+be\s+submitted\s+(?:via|through|using)\s+([^.]+?)(?:\.|$)',
                r'(?:submit|file)\s+claims?\s+(?:electronically|online)\s+(?:via|through|at)\s+([^.]+?)(?:\.|$)',
                r'(?:electronic\s+)?claim\s+submission:?\s*([^.]+?)(?:\.|$)',
            ],
            'eligibility': [
                r'(?:eligibility|coverage)\s+(?:verification|check)(?:\s+(?:is\s+)?required|:)\s*([^.]+?)(?:\.|$)',
                r'verify\s+(?:patient\s+)?eligibility\s+(?:before|prior\s+to)\s+([^.]+?)(?:\.|$)',
                r'(?:eligibility|coverage)\s+(?:must\s+be\s+)?(?:verified|confirmed)\s*([^.]+?)(?:\.|$)',
            ],
            'referrals': [
                r'referral(?:s)?\s+(?:are\s+)?(?:required|needed)\s+for\s+([^.]+?)(?:\.|$)',
                r'requires?\s+(?:a\s+)?referral:?\s*([^.]+?)(?:\.|$)',
                r'(?:specialist|specialty)\s+services\s+require\s+(?:a\s+)?referral',
            ],
            'documentation': [
                r'(?:documentation|records?|notes?)\s+(?:required|needed|must\s+include):?\s*([^.]+?)(?:\.|$)',
                r'(?:medical\s+)?(?:records?|documentation)\s+must\s+(?:include|contain)\s+([^.]+?)(?:\.|$)',
                r'submit\s+(?:the\s+)?following\s+documentation:?\s*([^.]+?)(?:\.|$)',
            ],
            'coding': [
                r'(?:ICD-10|CPT|HCPCS)\s+code(?:s)?\s+([^.]+?)(?:\.|$)',
                r'(?:diagnosis|procedure)\s+code(?:s)?\s+(?:required|must\s+be)\s+([^.]+?)(?:\.|$)',
                r'coding\s+(?:requirements?|guidelines?):?\s*([^.]+?)(?:\.|$)',
            ],
        }
        
        # Noise patterns to remove
        self.noise_patterns = [
            r'©\s*\d{4}.*?(?:All\s+rights\s+reserved|Confidential)',
            r'Page\s+\d+\s+of\s+\d+',
            r'Proprietary\s+and\s+Confidential',
            r'For\s+(?:Internal|Provider)\s+Use\s+Only',
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # Dates in headers/footers
        ]
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract high-quality structured data from healthcare PDF
        
        Returns:
            Dictionary with metadata, extracted rules, and clean text
        """
        print(f"\n{'='*70}")
        print(f"Processing: {Path(pdf_path).name}")
        print('='*70)
        
        # Try multiple extraction methods
        text = self._extract_text_hybrid(pdf_path)
        
        if not text or len(text) < 100:
            print("⚠️  Insufficient text extracted")
            return None
        
        # Clean the text
        clean_text = self._clean_text(text)
        
        # Extract structured rules
        rules = self._extract_rules(clean_text)
        
        # Extract metadata
        metadata = self._extract_metadata(pdf_path, clean_text)
        
        # Create sections
        sections = self._identify_sections(clean_text)
        
        result = {
            "metadata": metadata,
            "extracted_rules": rules,
            "sections": sections,
            "statistics": {
                "total_rules": len(rules),
                "rules_by_type": self._count_by_type(rules),
                "text_length": len(clean_text),
                "sections_found": len(sections),
            },
            "quality_metrics": self._assess_quality(rules, clean_text),
            "processing_info": {
                "processed_date": datetime.now().isoformat(),
                "extractor_version": "2.0"
            }
        }
        
        # Print summary
        print(f"\n📊 Extraction Summary:")
        print(f"   ✓ Total rules extracted: {len(rules)}")
        print(f"   ✓ Sections identified: {len(sections)}")
        print(f"   ✓ Text length: {len(clean_text):,} characters")
        print(f"   ✓ Quality score: {result['quality_metrics']['overall_score']:.1%}")
        
        for rule_type, count in result['statistics']['rules_by_type'].items():
            print(f"   • {rule_type}: {count}")
        
        return result
    
    def _extract_text_hybrid(self, pdf_path: str) -> str:
        """Use multiple methods to extract text"""
        text = ""
        
        # Method 1: pdfplumber (better for tables and layout)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            if text:
                print("✓ Extracted text using pdfplumber")
                return text
        except Exception as e:
            print(f"⚠️  pdfplumber failed: {e}")
        
        # Method 2: PyPDF2 (fallback)
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            if text:
                print("✓ Extracted text using PyPDF2")
                return text
        except Exception as e:
            print(f"❌ PyPDF2 failed: {e}")
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Remove noise and normalize text"""
        # Remove noise patterns
        for pattern in self.noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def _extract_rules(self, text: str) -> List[Dict[str, Any]]:
        """Extract structured rules using patterns"""
        rules = []
        
        # Split into sentences for better context
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for rule_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    # Get the full sentence for context
                    match_start = match.start()
                    sentence = self._get_sentence_at_position(text, match_start)
                    
                    if not sentence or len(sentence) < 20:
                        continue
                    
                    # Extract the specific requirement
                    requirement = match.group(1) if match.lastindex else match.group(0)
                    requirement = requirement.strip()
                    
                    # Skip if it's just a header or too vague
                    if self._is_valid_rule(requirement, sentence):
                        rules.append({
                            "rule_type": rule_type,
                            "requirement": requirement,
                            "full_context": sentence,
                            "confidence": self._calculate_confidence(sentence, rule_type),
                            "source_text": match.group(0)
                        })
        
        # Deduplicate similar rules
        rules = self._deduplicate_rules(rules)
        
        return rules
    
    def _get_sentence_at_position(self, text: str, position: int) -> str:
        """Get the full sentence containing a position"""
        # Find sentence boundaries
        start = text.rfind('.', 0, position)
        if start == -1:
            start = 0
        else:
            start += 1
        
        end = text.find('.', position)
        if end == -1:
            end = len(text)
        else:
            end += 1
        
        sentence = text[start:end].strip()
        return sentence
    
    def _is_valid_rule(self, requirement: str, context: str) -> bool:
        """Check if extracted text is a valid rule"""
        # Skip headers and section titles
        if len(requirement) < 10:
            return False
        
        # Skip if it's all uppercase (likely a header)
        if requirement.isupper() and len(requirement) > 30:
            return False
        
        # Skip if it's mostly numbers
        if sum(c.isdigit() for c in requirement) > len(requirement) * 0.5:
            return False
        
        # Must contain some meaningful content
        meaningful_words = ['required', 'must', 'need', 'submit', 'within', 
                           'authorization', 'claim', 'file', 'verify']
        if not any(word in context.lower() for word in meaningful_words):
            return False
        
        return True
    
    def _calculate_confidence(self, sentence: str, rule_type: str) -> float:
        """Calculate confidence score for extracted rule"""
        confidence = 0.5
        
        # Higher confidence for specific keywords
        high_confidence_words = ['must', 'required', 'shall', 'will']
        if any(word in sentence.lower() for word in high_confidence_words):
            confidence += 0.2
        
        # Higher confidence for numeric specifics (deadlines, etc.)
        if re.search(r'\d+\s+(?:days?|months?|years?)', sentence):
            confidence += 0.15
        
        # Higher confidence for complete sentences
        if sentence.count('.') == 1 and len(sentence.split()) > 8:
            confidence += 0.1
        
        # Type-specific confidence
        if rule_type == 'timely_filing' and re.search(r'\d+', sentence):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _deduplicate_rules(self, rules: List[Dict]) -> List[Dict]:
        """Remove duplicate or very similar rules"""
        unique_rules = []
        seen_requirements = set()
        
        for rule in rules:
            # Normalize for comparison
            normalized = re.sub(r'\s+', ' ', rule['requirement'].lower()).strip()
            
            # Check if we've seen something very similar
            is_duplicate = False
            for seen in seen_requirements:
                similarity = self._similarity_ratio(normalized, seen)
                if similarity > 0.85:  # 85% similar
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_rules.append(rule)
                seen_requirements.add(normalized)
        
        return unique_rules
    
    def _similarity_ratio(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        # Simple word-based similarity
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _extract_metadata(self, pdf_path: str, text: str) -> Dict[str, Any]:
        """Extract metadata from PDF"""
        metadata = {
            "filename": Path(pdf_path).name,
            "file_size": Path(pdf_path).stat().st_size,
            "extraction_date": datetime.now().isoformat(),
        }
        
        # Try to identify payer
        payers = {
            'ANTHEM': ['anthem', 'elevance'],
            'UHC': ['united healthcare', 'unitedhealthcare', 'uhc'],
            'AETNA': ['aetna', 'cvs health'],
            'CIGNA': ['cigna', 'evernorth'],
            'HUMANA': ['humana'],
            'BCBS': ['blue cross', 'blue shield', 'bcbs'],
        }
        
        text_lower = text[:5000].lower()  # Check first 5000 chars
        for payer, keywords in payers.items():
            if any(keyword in text_lower for keyword in keywords):
                metadata['payer'] = payer
                break
        
        # Try to extract state
        states = ['alabama', 'alaska', 'arizona', 'california', 'florida', 
                 'georgia', 'illinois', 'maine', 'massachusetts', 'new york',
                 'texas', 'washington']
        for state in states:
            if state in text_lower:
                metadata['state'] = state.title()
                break
        
        # Try to extract effective date
        date_pattern = r'effective\s+(?:date:?\s*)?(\w+\s+\d{1,2},?\s+\d{4})'
        date_match = re.search(date_pattern, text[:2000], re.IGNORECASE)
        if date_match:
            metadata['effective_date'] = date_match.group(1)
        
        return metadata
    
    def _identify_sections(self, text: str) -> List[Dict[str, str]]:
        """Identify major sections in the document"""
        sections = []
        
        # Common section headers
        section_patterns = [
            r'(?:^|\n)([A-Z][A-Z\s]{15,})(?:\n|$)',  # ALL CAPS headers
            r'(?:^|\n)((?:SECTION|CHAPTER)\s+\d+[:.]\s*[A-Z][^.]+)(?:\n|$)',
            r'(?:^|\n)(\d+\.\s+[A-Z][^.]+)(?:\n|$)',  # Numbered sections
        ]
        
        for pattern in section_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                section_title = match.group(1).strip()
                if 10 < len(section_title) < 100:
                    sections.append({
                        "title": section_title,
                        "position": match.start()
                    })
        
        # Sort by position
        sections.sort(key=lambda x: x['position'])
        
        # Remove position info
        for section in sections:
            del section['position']
        
        return sections[:20]  # Limit to 20 sections
    
    def _count_by_type(self, rules: List[Dict]) -> Dict[str, int]:
        """Count rules by type"""
        counts = {}
        for rule in rules:
            rule_type = rule['rule_type']
            counts[rule_type] = counts.get(rule_type, 0) + 1
        return counts
    
    def _assess_quality(self, rules: List[Dict], text: str) -> Dict[str, Any]:
        """Assess the quality of extraction"""
        total_rules = len(rules)
        
        # Calculate average confidence
        avg_confidence = sum(r['confidence'] for r in rules) / total_rules if total_rules > 0 else 0
        
        # Check diversity of rule types
        rule_types = len(set(r['rule_type'] for r in rules))
        
        # Overall quality score
        quality_score = 0.0
        if total_rules > 0:
            quality_score += min(total_rules / 50, 0.4)  # More rules = better (up to 0.4)
        quality_score += min(avg_confidence, 0.3)  # Confidence (up to 0.3)
        quality_score += min(rule_types / 8, 0.3)  # Diversity (up to 0.3)
        
        return {
            "overall_score": quality_score,
            "average_confidence": avg_confidence,
            "rule_types_found": rule_types,
            "total_rules_extracted": total_rules,
            "text_quality": "good" if len(text) > 5000 else "low"
        }


# ============================================================================
# BATCH PROCESSOR
# ============================================================================

class BatchPDFProcessor:
    """Process multiple PDFs and save high-quality JSON"""
    
    def __init__(self, output_dir: str = "./high_quality_json"):
        self.extractor = HealthcarePolicyExtractor()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def process_directory(self, pdf_dir: str) -> List[Dict]:
        """Process all PDFs in a directory"""
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        
        print(f"\n{'='*70}")
        print(f"Found {len(pdf_files)} PDF files to process")
        print('='*70)
        
        results = []
        successful = 0
        failed = 0
        
        for pdf_file in pdf_files:
            try:
                result = self.extractor.extract_from_pdf(str(pdf_file))
                
                if result:
                    # Save individual JSON
                    output_file = self.output_dir / f"{pdf_file.stem}.json"
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    results.append(result)
                    successful += 1
                    print(f"✅ Saved: {output_file.name}")
                else:
                    failed += 1
                    print(f"❌ Failed: {pdf_file.name}")
                    
            except Exception as e:
                failed += 1
                print(f"❌ Error processing {pdf_file.name}: {e}")
        
        # Save summary
        summary = {
            "processing_date": datetime.now().isoformat(),
            "total_files": len(pdf_files),
            "successful": successful,
            "failed": failed,
            "files_processed": [r['metadata']['filename'] for r in results]
        }
        
        summary_file = self.output_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING COMPLETE")
        print('='*70)
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📁 Output directory: {self.output_dir}")
        
        return results


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Single file processing
    if len(sys.argv) > 1 and sys.argv[1].endswith('.pdf'):
        extractor = HealthcarePolicyExtractor()
        result = extractor.extract_from_pdf(sys.argv[1])
        
        # Save output
        output_file = f"{Path(sys.argv[1]).stem}_extracted.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✅ Saved to: {output_file}")
    
    # Batch processing
    else:
        pdf_directory = sys.argv[1] if len(sys.argv) > 1 else "./payer_pdfs"
        
        processor = BatchPDFProcessor(output_dir="./high_quality_json")
        results = processor.process_directory(pdf_directory)
        
        print(f"\n✨ Processing complete! Check ./high_quality_json/ for results")