"""
Healthcare Policy Deduplication and Temporal Management System
Integrates with existing Azure Blob Storage for BIG_KnowledgeBase project
"""

import re
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import anthropic
from azure.storage.blob import BlobServiceClient
from difflib import SequenceMatcher
import logging


@dataclass
class PolicyMetadata:
    """Structured metadata for healthcare policies"""
    policy_id: str
    policy_hash: str  # Content-based identifier
    policy_type: str  # prior_auth, timely_filing, appeals, etc.
    effective_date: Optional[datetime]
    end_date: Optional[datetime]
    document_date: Optional[datetime]
    version: Optional[str]
    payer_name: str
    source_pdf: str
    source_url: str
    crawl_timestamp: datetime
    confidence_score: float  # 0-1, how confident are we in the extraction
    content: Dict
    explicit_id: Optional[str] = None  # If policy has explicit ID
    supersedes: Optional[List[str]] = None  # IDs of policies this replaces


class PolicyIDExtractor:
    """
    Hybrid approach to extract policy identifiers:
    1. Try to extract explicit policy IDs (codes, numbers)
    2. Fall back to content-based similarity for unstructured policies
    """
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
        
        # Common policy ID patterns in healthcare
        self.id_patterns = [
            r'Policy\s*(?:Number|#|ID|Code)[\s:]*([A-Z0-9\-]+)',
            r'(?:PA|AUTH|CLM|APL)[\s\-]?(\d{3,})',  # PA-001, AUTH123
            r'(?:CPT|HCPCS)\s*Code[\s:]*(\d{4,5})',
            r'Procedure\s*Code[\s:]*([A-Z0-9]+)',
            r'Reference\s*Number[\s:]*([A-Z0-9\-]+)',
            r'Document\s*ID[\s:]*([A-Z0-9\-]+)',
        ]
        
        # Date extraction patterns
        self.date_patterns = [
            (r'Effective\s*Date[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'effective'),
            (r'Effective[\s:]*(\w+\s+\d{1,2},?\s+\d{4})', 'effective'),
            (r'Valid\s*(?:From|Through)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'effective'),
            (r'Exp(?:iration|ires?)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'end'),
            (r'End\s*Date[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'end'),
            (r'Revised[\s:]*(\w+\s+\d{1,2},?\s+\d{4})', 'document'),
            (r'Updated[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'document'),
            (r'Supersedes.*?(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 'supersedes'),
        ]
    
    def extract_explicit_id(self, text: str) -> Optional[str]:
        """Extract explicit policy ID if present"""
        for pattern in self.id_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def extract_dates(self, text: str) -> Dict[str, Optional[datetime]]:
        """Extract all relevant dates from policy text"""
        dates = {
            'effective_date': None,
            'end_date': None,
            'document_date': None
        }
        
        for pattern, date_type in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(1)
                parsed_date = self._parse_date(date_str)
                if parsed_date and not dates[f'{date_type}_date']:
                    dates[f'{date_type}_date'] = parsed_date
        
        return dates
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats"""
        date_formats = [
            '%m/%d/%Y', '%m-%d-%Y', '%m/%d/%y', '%m-%d-%y',
            '%Y-%m-%d', '%Y/%m/%d',
            '%B %d, %Y', '%B %d %Y', '%b %d, %Y', '%b %d %Y'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        return None
    
    def generate_content_hash(self, content: Dict) -> str:
        """Generate hash from policy content for similarity matching"""
        # Normalize key content fields
        normalizable = {
            'policy_type': content.get('policy_type', ''),
            'payer': content.get('payer_name', ''),
            'procedures': sorted(content.get('procedures', [])),
            'requirements': sorted(content.get('requirements', [])),
        }
        
        # Create stable hash
        content_str = json.dumps(normalizable, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def extract_supersession_info(self, text: str) -> List[str]:
        """Extract information about which policies this replaces"""
        supersedes_patterns = [
            r'[Ss]upersedes\s*(?:Policy\s*)?([A-Z0-9\-,\s]+)',
            r'[Rr]eplaces\s*(?:Policy\s*)?([A-Z0-9\-,\s]+)',
            r'[Oo]bsoletes\s*(?:Policy\s*)?([A-Z0-9\-,\s]+)',
        ]
        
        superseded_ids = []
        for pattern in supersedes_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                ids_str = match.group(1)
                # Split by comma and extract IDs
                ids = re.findall(r'[A-Z0-9\-]+', ids_str)
                superseded_ids.extend(ids)
        
        return list(set(superseded_ids))
    
    def calculate_similarity(self, content1: Dict, content2: Dict) -> float:
        """Calculate content similarity between two policies"""
        # Compare key fields
        str1 = json.dumps(content1, sort_keys=True)
        str2 = json.dumps(content2, sort_keys=True)
        
        return SequenceMatcher(None, str1, str2).ratio()
    
    async def llm_extract_policy_id(self, text: str) -> Optional[Dict]:
        """Use Claude to extract policy metadata when patterns fail"""
        if not self.client:
            return None
        
        prompt = f"""Extract policy metadata from this healthcare policy text:

{text[:2000]}  # Limit to first 2000 chars

Return ONLY a JSON object with these fields:
- policy_id: explicit policy identifier if found
- policy_type: type of policy (prior_auth, timely_filing, appeals, claims, etc.)
- effective_date: when policy takes effect (YYYY-MM-DD format)
- end_date: when policy expires (YYYY-MM-DD format) 
- supersedes: list of policy IDs this replaces
- confidence: your confidence 0-1 in these extractions

If a field is not found, use null. Be conservative - only extract if clearly stated."""

        try:
            message = await self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text
            # Strip markdown fences if present
            cleaned = re.sub(r'```json\n?|```', '', response_text).strip()
            return json.loads(cleaned)
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return None


class PolicyDeduplicationEngine:
    """
    Main engine for managing policy versions and removing expired rules
    """
    
    def __init__(self, azure_connection_string: str, container_name: str = "healthcare-policies"):
        self.blob_service = BlobServiceClient.from_connection_string(azure_connection_string)
        self.container_name = container_name
        self.extractor = PolicyIDExtractor()
        self.logger = logging.getLogger(__name__)
        
        # Ensure containers exist
        self._ensure_containers()
    
    def _ensure_containers(self):
        """Create necessary blob containers"""
        containers = [
            self.container_name,  # Active policies
            f"{self.container_name}-archive",  # Expired policies
            f"{self.container_name}-metadata"  # Indexes and metadata
        ]
        
        for container in containers:
            try:
                self.blob_service.create_container(container)
            except Exception:
                pass  # Container already exists
    
    def process_extracted_policy(self, 
                                 content: Dict, 
                                 raw_text: str,
                                 source_pdf: str,
                                 source_url: str,
                                 payer_name: str) -> PolicyMetadata:
        """
        Process a newly extracted policy and create metadata
        Implements the Hybrid approach (Option C)
        """
        
        # Step 1: Try to extract explicit policy ID
        explicit_id = self.extractor.extract_explicit_id(raw_text)
        
        # Step 2: Extract dates
        dates = self.extractor.extract_dates(raw_text)
        
        # Step 3: Generate content-based hash
        content_hash = self.extractor.generate_content_hash(content)
        
        # Step 4: Determine policy_id (hybrid approach)
        if explicit_id:
            policy_id = f"{payer_name}_{explicit_id}"
            confidence = 0.95
        else:
            # Fall back to content hash
            policy_id = f"{payer_name}_{content_hash}"
            confidence = 0.7
        
        # Step 5: Extract supersession info
        supersedes = self.extractor.extract_supersession_info(raw_text)
        
        # Step 6: Confidence adjustment
        if not dates['effective_date']:
            confidence *= 0.8  # Lower confidence if no dates
        
        # Create metadata object
        metadata = PolicyMetadata(
            policy_id=policy_id,
            policy_hash=content_hash,
            policy_type=content.get('policy_type', 'unknown'),
            effective_date=dates['effective_date'],
            end_date=dates['end_date'],
            document_date=dates['document_date'],
            version=content.get('version'),
            payer_name=payer_name,
            source_pdf=source_pdf,
            source_url=source_url,
            crawl_timestamp=datetime.now(),
            confidence_score=confidence,
            content=content,
            explicit_id=explicit_id,
            supersedes=supersedes if supersedes else None
        )
        
        return metadata
    
    def deduplicate_policies(self, policies: List[PolicyMetadata]) -> List[PolicyMetadata]:
        """
        Given a list of policies, identify and keep only current versions
        """
        
        # Group by policy_id
        policy_groups = defaultdict(list)
        for policy in policies:
            policy_groups[policy.policy_id].append(policy)
        
        current_policies = []
        reference_date = datetime.now()
        
        for policy_id, versions in policy_groups.items():
            # Sort by effective_date (most recent first), then by crawl_timestamp
            versions.sort(
                key=lambda p: (
                    p.effective_date or datetime.min,
                    p.crawl_timestamp
                ),
                reverse=True
            )
            
            # Find the most recent valid policy
            for policy in versions:
                if self._is_policy_current(policy, reference_date):
                    current_policies.append(policy)
                    break  # Only keep the most recent valid version
        
        return current_policies
    
    def _is_policy_current(self, policy: PolicyMetadata, reference_date: datetime) -> bool:
        """Check if a policy is currently valid"""
        
        # If no dates, use crawl timestamp heuristic
        if not policy.effective_date:
            # Consider recent crawls (last 2 years) as potentially current
            cutoff = reference_date - timedelta(days=730)
            return policy.crawl_timestamp >= cutoff
        
        # Check effective date
        if policy.effective_date > reference_date:
            return False  # Future policy
        
        # Check end date
        if policy.end_date and policy.end_date < reference_date:
            return False  # Expired
        
        return True
    
    def find_similar_policies(self, policy: PolicyMetadata, 
                             all_policies: List[PolicyMetadata],
                             threshold: float = 0.85) -> List[PolicyMetadata]:
        """
        Find policies that might be the same rule despite different IDs
        Used for cross-checking and manual review
        """
        similar = []
        
        for other in all_policies:
            if other.policy_id == policy.policy_id:
                continue
            
            if other.payer_name != policy.payer_name:
                continue  # Different payers
            
            similarity = self.extractor.calculate_similarity(
                policy.content, 
                other.content
            )
            
            if similarity >= threshold:
                similar.append(other)
        
        return similar
    
    def save_to_azure(self, policy: PolicyMetadata, container_suffix: str = ""):
        """Save policy to Azure Blob Storage"""
        container = f"{self.container_name}{container_suffix}"
        blob_name = f"{policy.payer_name}/{policy.policy_id}.json"
        
        blob_client = self.blob_service.get_blob_client(
            container=container,
            blob=blob_name
        )
        
        # Convert to JSON
        data = asdict(policy)
        # Handle datetime serialization
        for key in ['effective_date', 'end_date', 'document_date', 'crawl_timestamp']:
            if data[key]:
                data[key] = data[key].isoformat()
        
        blob_client.upload_blob(
            json.dumps(data, indent=2),
            overwrite=True
        )

    def _deserialize_policy(self, data: Dict) -> PolicyMetadata:
        """Convert stored JSON back into PolicyMetadata"""
        for key in ['effective_date', 'end_date', 'document_date', 'crawl_timestamp']:
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
            else:
                data[key] = None
        return PolicyMetadata(**data)

    def _load_policies_for_payer(self, payer_name: str) -> List[PolicyMetadata]:
        """Load all policies for a payer from the active container"""
        container_client = self.blob_service.get_container_client(self.container_name)
        policies = []

        prefix = f"{payer_name}/"
        for blob in container_client.list_blobs(name_starts_with=prefix):
            blob_client = self.blob_service.get_blob_client(
                container=self.container_name,
                blob=blob.name
            )
            try:
                data = json.loads(blob_client.download_blob().readall())
                policies.append(self._deserialize_policy(data))
            except Exception as e:
                self.logger.warning(f"Failed to read blob {blob.name}: {e}")
        return policies

    def _delete_blob(self, blob_name: str):
        """Delete a blob from the active container"""
        try:
            container_client = self.blob_service.get_container_client(self.container_name)
            container_client.delete_blob(blob_name)
            self.logger.info(f"Deleted expired/replaced policy blob: {blob_name}")
        except Exception as e:
            self.logger.warning(f"Failed to delete blob {blob_name}: {e}")

    def remove_replaced_policies(self, new_policy: PolicyMetadata):
        """
        Remove expired policies for the same payer + policy_type that are superseded by the new one.
        """
        if not new_policy.payer_name or not new_policy.policy_type:
            return

        existing = self._load_policies_for_payer(new_policy.payer_name)
        now = datetime.now()

        for old in existing:
            if old.policy_id == new_policy.policy_id:
                continue
            if old.policy_type != new_policy.policy_type:
                continue

            # Determine if old policy should be removed
            old_expired = old.end_date and old.end_date < now
            superseded_by_date = (
                old.end_date and new_policy.effective_date and old.end_date < new_policy.effective_date
            )

            if old_expired or superseded_by_date:
                blob_name = f"{old.payer_name}/{old.policy_id}.json"
                self._delete_blob(blob_name)
    
    def load_all_policies(self) -> List[PolicyMetadata]:
        """Load all policies from Azure storage"""
        container_client = self.blob_service.get_container_client(self.container_name)
        policies = []
        
        for blob in container_client.list_blobs():
            blob_client = self.blob_service.get_blob_client(
                container=self.container_name,
                blob=blob.name
            )
            
            data = json.loads(blob_client.download_blob().readall())
            
            # Reconstruct PolicyMetadata
            for key in ['effective_date', 'end_date', 'document_date', 'crawl_timestamp']:
                if data[key]:
                    data[key] = datetime.fromisoformat(data[key])
            
            policies.append(PolicyMetadata(**data))
        
        return policies
    
    def cleanup_expired_policies(self) -> Dict:
        """
        Main cleanup job: Remove expired policies and keep only current ones
        Returns statistics about the cleanup
        """
        
        print("Loading all policies from Azure...")
        all_policies = self.load_all_policies()
        
        print(f"Found {len(all_policies)} total policies")
        
        print("Deduplicating and identifying current policies...")
        current_policies = self.deduplicate_policies(all_policies)
        
        print(f"Identified {len(current_policies)} current policies")
        
        # Identify expired policies
        all_ids = {p.policy_id for p in all_policies}
        current_ids = {p.policy_id for p in current_policies}
        expired_ids = all_ids - current_ids
        
        expired_policies = [p for p in all_policies if p.policy_id in expired_ids]
        
        print(f"Archiving {len(expired_policies)} expired policies...")
        
        # Archive expired policies
        for policy in expired_policies:
            self.save_to_azure(policy, container_suffix="-archive")
        
        # Update active knowledge base with only current policies
        print("Updating active knowledge base...")
        container_client = self.blob_service.get_container_client(self.container_name)
        
        # Clear current container
        for blob in container_client.list_blobs():
            container_client.delete_blob(blob.name)
        
        # Save only current policies
        for policy in current_policies:
            self.save_to_azure(policy)
        
        # Generate and save metadata index
        self._save_policy_index(current_policies)
        
        stats = {
            'total_policies_processed': len(all_policies),
            'current_policies': len(current_policies),
            'archived_policies': len(expired_policies),
            'cleanup_timestamp': datetime.now().isoformat(),
            'policies_by_payer': self._count_by_payer(current_policies),
            'low_confidence_policies': len([p for p in current_policies if p.confidence_score < 0.7])
        }
        
        print(f"\nCleanup complete! Stats:")
        print(json.dumps(stats, indent=2))
        
        return stats
    
    def _count_by_payer(self, policies: List[PolicyMetadata]) -> Dict[str, int]:
        """Count policies by payer"""
        counts = defaultdict(int)
        for policy in policies:
            counts[policy.payer_name] += 1
        return dict(counts)
    
    def _save_policy_index(self, policies: List[PolicyMetadata]):
        """Save an index of all current policies for fast lookup"""
        index = {
            'last_updated': datetime.now().isoformat(),
            'total_policies': len(policies),
            'policies': [
                {
                    'policy_id': p.policy_id,
                    'payer': p.payer_name,
                    'type': p.policy_type,
                    'effective_date': p.effective_date.isoformat() if p.effective_date else None,
                    'confidence': p.confidence_score
                }
                for p in policies
            ]
        }
        
        blob_client = self.blob_service.get_blob_client(
            container=f"{self.container_name}-metadata",
            blob="policy_index.json"
        )
        
        blob_client.upload_blob(
            json.dumps(index, indent=2),
            overwrite=True
        )


# Example usage
if __name__ == "__main__":
    # Initialize the deduplication engine
    AZURE_CONNECTION_STRING = "your_azure_connection_string"
    
    engine = PolicyDeduplicationEngine(AZURE_CONNECTION_STRING)
    
    # Example: Process a newly extracted policy
    sample_content = {
        "policy_type": "prior_authorization",
        "payer_name": "anthem",
        "procedures": ["CPT 99213", "CPT 99214"],
        "requirements": ["Prior auth required within 48 hours"],
        "description": "Office visit prior authorization requirements"
    }
    
    sample_text = """
    Policy Number: PA-2024-001
    Effective Date: 01/01/2024
    End Date: 12/31/2024
    
    Prior Authorization Requirements for Office Visits
    This policy supersedes PA-2023-001
    """
    
    # Process the policy (Hybrid Option C)
    metadata = engine.process_extracted_policy(
        content=sample_content,
        raw_text=sample_text,
        source_pdf="anthem_policies_2024.pdf",
        source_url="https://anthem.com/provider/policies",
        payer_name="anthem"
    )
    
    print(f"Extracted Policy ID: {metadata.policy_id}")
    print(f"Confidence Score: {metadata.confidence_score}")
    print(f"Effective Date: {metadata.effective_date}")
    
    # Save to Azure
    engine.save_to_azure(metadata)
    
    # Run cleanup job (removes expired policies)
    stats = engine.cleanup_expired_policies()
