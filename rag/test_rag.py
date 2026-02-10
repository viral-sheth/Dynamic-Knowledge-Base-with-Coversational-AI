"""
Healthcare Knowledge Base RAG Testing System
Tests RAG implementation with your actual extracted healthcare rules
"""


import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Local embedding model (no Azure): BAAI/bge-small-en-v1.5
# Install once with: pip install sentence-transformers numpy
try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise ImportError(
        "sentence-transformers is not installed. "
        "Run: pip install sentence-transformers numpy"
    ) from exc


class HealthcareRAG:
    """RAG system for healthcare payer rules"""
    
    def __init__(self, json_file: str):
        print("🔧 Initializing Healthcare RAG System...")
        # Local embedding model from Hugging Face (load from local cache only)
        # Make sure you've downloaded it once with:
        #   from sentence_transformers import SentenceTransformer
        #   SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5", local_files_only=True)

        json_path = self.resolve_json_file(json_file)
        self.data = self.load_data(json_path)
        self.rules = self.data.get('healthcare_rules', [])
        self.metadata = self.data.get('metadata', {})
        self.summary = self.data.get('summary', {})
        
        # Make missing summary/metadata fields not crash when using sample JSON
        payer_names = {r.get('payer_name') for r in self.rules if r.get('payer_name')}
        filenames = {r.get('filename') for r in self.rules if r.get('filename')}
        rule_types = {r.get('rule_type') for r in self.rules if r.get('rule_type')}
        
        self.metadata.setdefault('total_rules', len(self.rules))
        self.metadata.setdefault('unique_payers', len(payer_names))
        self.metadata.setdefault('unique_documents', len(filenames))
        self.summary.setdefault('rule_types', sorted(rule_types))
        
        print(f"📊 Loaded {len(self.rules)} rules from {self.metadata['unique_payers']} payers")
        print(f"📋 Rule types: {', '.join(self.summary['rule_types'])}")
        
        self.embeddings = self.create_embeddings()
        print("✅ RAG system ready!\n")
    
    def resolve_json_file(self, json_file: str) -> str:
        """Return the JSON path, falling back to sample_rules.json if needed"""
        path = Path(json_file)
        if path.exists():
            return str(path)
        
        fallback = Path('sample_rules.json')
        if fallback.exists():
            print(f"⚠️ {json_file} not found. Using {fallback.name} instead.")
            return str(fallback)
        
        raise FileNotFoundError(f"Neither {json_file} nor {fallback} found.")
    
    def load_data(self, json_file: str) -> Dict:
        """Load healthcare rules from JSON file"""
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_embeddings(self):
        """Create embeddings for all rule contents"""
        print("🧠 Creating embeddings for rules...")
        texts = [
            f"{rule['rule_title']}: {rule['rule_content']}" 
            for rule in self.rules
        ]
        return self.embed_texts(texts)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts using the local sentence-transformers model."""
        if not texts:
            return np.array([])
        # convert_to_numpy=True gives a NumPy array directly
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
    
    def search(self, query: str, top_k: int = 5, filter_by: Dict = None) -> List[Dict]:
        """
        Search for relevant rules
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_by: Optional filters like {'rule_type': 'appeals', 'payer_name': 'United Healthcare'}
        """
        # Encode the query
        query_embedding = self.embed_texts([query])[0]
        
        # Filter rules if requested
        filtered_rules = self.rules
        filtered_embeddings = self.embeddings
        
        if filter_by:
            filtered_indices = []
            for idx, rule in enumerate(self.rules):
                match = True
                for key, value in filter_by.items():
                    if rule.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_indices.append(idx)
            
            filtered_rules = [self.rules[i] for i in filtered_indices]
            filtered_embeddings = self.embeddings[filtered_indices]
            
            if len(filtered_rules) == 0:
                print(f"⚠️ No rules found matching filters: {filter_by}")
                return []
        
        # Calculate cosine similarity
        similarities = np.dot(filtered_embeddings, query_embedding) / (
            np.linalg.norm(filtered_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'rule': filtered_rules[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def format_answer(self, query: str, results: List[Dict]) -> str:
        """Format search results into a concise, cited answer."""
        if not results:
            return "No relevant rules found for your query."

        lines = [f"Answer based on top {len(results)} rule(s):", "(See linked sources below for documents)"]
        for i, result in enumerate(results, 1):
            rule = result["rule"]
            title = rule.get("rule_title") or "Policy rule"
            payer = rule.get("payer_name") or "Unknown payer"
            scope = rule.get("geographic_scope") or "N/A"
            snippet = (rule.get("rule_content") or "").strip()
            if len(snippet) > 320:
                snippet = snippet[:320].rstrip() + "..."

            lines.append(f"{i}) {title} — {payer} ({scope})")
            if snippet:
                lines.append(f"    {snippet}")

        return "\n".join(lines)
    
    def get_statistics(self) -> str:
        """Get statistics about the knowledge base"""
        stats = f"""
📊 KNOWLEDGE BASE STATISTICS
{'=' * 80}

Total Rules: {self.metadata['total_rules']}
Unique Payers: {self.metadata['unique_payers']}
Documents Processed: {self.metadata.get('unique_documents', 'N/A')}
Export Date: {self.metadata['export_timestamp']}

RULES BY TYPE:
"""
        for rule_type, count in self.summary['rules_by_type'].items():
            stats += f"  • {rule_type.title()}: {count}\n"
        
        stats += f"\nRULES BY PAYER:\n"
        for payer, count in self.summary['rules_by_payer'].items():
            stats += f"  • {payer}: {count}\n"
        
        stats += f"\nDOCUMENTS:\n"
        for doc, count in self.summary['rules_by_document'].items():
            stats += f"  • {doc}: {count} rules\n"
        
        return stats
    
    def analyze_payer(self, payer_name: str) -> str:
        """Analyze rules for a specific payer"""
        payer_rules = [r for r in self.rules if r['payer_name'] == payer_name]
        
        if not payer_rules:
            return f"❌ No rules found for payer: {payer_name}"
        
        # Count by type
        type_counts = defaultdict(int)
        for rule in payer_rules:
            type_counts[rule['rule_type']] += 1
        
        analysis = f"""
🏥 PAYER ANALYSIS: {payer_name}
{'=' * 80}

Total Rules: {len(payer_rules)}

Rules by Type:
"""
        for rule_type, count in sorted(type_counts.items()):
            analysis += f"  • {rule_type.title()}: {count}\n"
        
        # Documents
        docs = set(r['filename'] for r in payer_rules)
        analysis += f"\nDocuments: {len(docs)}\n"
        for doc in docs:
            analysis += f"  • {doc}\n"
        
        return analysis

    # ──────────────────────────────────────────────────────────────────────────
    # LLM ANSWER GENERATION
    # ──────────────────────────────────────────────────────────────────────────

    def build_context_from_results(self, results: List[Dict]) -> str:
        """Turn retrieved rules into a compact text context for the LLM."""
        if not results:
            return "No relevant rules were retrieved from the knowledge base."

        chunks = []
        for r in results:
            rule = r["rule"]
            chunks.append(
                f"Payer: {rule.get('payer_name')}\n"
                f"Rule Type: {rule.get('rule_type')}\n"
                f"Title: {rule.get('rule_title')}\n"
                f"Document: {rule.get('filename')} (page {rule.get('page_number')})\n"
                f"Content: {rule.get('rule_content')}\n"
            )
        return "\n\n---\n\n".join(chunks)

    def generate_llm_answer(
        self,
        query: str,
        results: List[Dict],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a natural-language answer using a Groq-hosted LLM
        (e.g., Llama 3.1 8B) on top of the retrieved rules.

        Requires:
          - pip install groq
          - GROQ_API_KEY set in your environment (or .env)
          - Optional: GROQ_MODEL to override the default model name
        """
        context = self.build_context_from_results(results)

        default_system_prompt = (
            "You are an assistant that answers questions about healthcare "
            "payer rules and policies. Use only the provided context. If the "
            "answer is not clearly supported by the context, say you are not sure."
        )
        system_prompt = system_prompt or default_system_prompt

        try:
            from groq import Groq
        except ImportError:
            return (
                "LLM answer not available: the 'groq' package is not installed.\n"
                "Install it with: pip install groq"
            )

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return (
                "LLM answer not available: GROQ_API_KEY is not set.\n"
                "Add GROQ_API_KEY to your environment or .env file."
            )

        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

        try:
            client = Groq(api_key=api_key)

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{query}\n\n"
                        f"Context from healthcare rules:\n{context}"
                    ),
                },
            ]

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=512,
            )

            return completion.choices[0].message.content.strip()
        except Exception as exc:
            return f"Error while calling Groq LLM: {exc}"


def run_tests():
    """Run comprehensive RAG tests"""
    
    print("🧪 HEALTHCARE RAG TESTING SUITE")
    print("=" * 80)
    print()
    interactive = sys.stdin.isatty()
    
    # Initialize RAG
    rag = HealthcareRAG('healthcare_rules_export.json')
    
    # Print statistics
    print(rag.get_statistics())
    print()
    
    # Test queries
    test_queries = [
        {
            'query': 'What are the timely filing requirements?',
            'filter': None
        },
        {
            'query': 'How do I file an appeal for a denied claim?',
            'filter': None
        },
        {
            'query': 'What is the prior authorization process?',
            'filter': None
        },
        {
            'query': 'Claims payment requirements',
            'filter': {'payer_name': 'United Healthcare'}
        },
        {
            'query': 'Appeals process',
            'filter': {'rule_type': 'appeals', 'payer_name': 'CountyCare Health Plan'}
        },
        {
            'query': 'What is the reimbursement policy?',
            'filter': None
        }
    ]
    
    print("\n🔍 RUNNING TEST QUERIES")
    print("=" * 80)
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n\n{'#' * 80}")
        print(f"TEST {i}/{len(test_queries)}")
        print(f"{'#' * 80}\n")
        
        if test['filter']:
            print(f"🔎 Filter: {test['filter']}")
        
        results = rag.search(test['query'], top_k=3, filter_by=test['filter'])
        answer = rag.format_answer(test['query'], results)
        print(answer)

        llm_answer = rag.generate_llm_answer(test['query'], results)
        print("🤖 LLM ANSWER")
        print("-" * 80)
        print(llm_answer)
        print("-" * 80 + "\n")
        if interactive:
            input("Press Enter to continue to next test...")
    
    # Payer analysis
    print("\n\n🏥 PAYER-SPECIFIC ANALYSIS")
    print("=" * 80)
    
    for payer in rag.summary['rules_by_payer'].keys():
        print(rag.analyze_payer(payer))
        print()
    
    # Interactive mode
    if not interactive:
        print("\n(Interactive mode skipped: non-interactive environment)")
        return

    print("\n\n💬 INTERACTIVE MODE")
    print("=" * 80)
    print("Ask questions about healthcare payer rules (type 'quit' to exit)")
    print("Example: 'What are United Healthcare's prior authorization requirements?'\n")
    
    while True:
        query = input("\n❓ Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            continue
        
        # Ask if they want to filter
        filter_choice = input("Filter by payer or type? (press Enter to skip): ").strip()
        filter_by = None
        
        if filter_choice:
            if 'united' in filter_choice.lower():
                filter_by = {'payer_name': 'United Healthcare'}
            elif 'county' in filter_choice.lower():
                filter_by = {'payer_name': 'CountyCare Health Plan'}
            elif 'appeal' in filter_choice.lower():
                filter_by = {'rule_type': 'appeals'}
            elif 'claim' in filter_choice.lower():
                filter_by = {'rule_type': 'claims'}
            elif 'prior' in filter_choice.lower():
                filter_by = {'rule_type': 'prior_authorization'}
            elif 'timely' in filter_choice.lower():
                filter_by = {'rule_type': 'timely_filing'}
        
        results = rag.search(query, top_k=3, filter_by=filter_by)
        answer = rag.format_answer(query, results)
        print("\n" + answer)

        llm_answer = rag.generate_llm_answer(query, results)
        print("🤖 LLM ANSWER")
        print("-" * 80)
        print(llm_answer)
        print("-" * 80)


if __name__ == "__main__":
    run_tests()
