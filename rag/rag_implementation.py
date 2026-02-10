"""
RAG Implementation for Healthcare Policy Bot
Processes PDFs, creates embeddings, and enables semantic search
"""

import os
import json
import numpy as np
from pathlib import Path
import re
from typing import List, Dict, Any
import anthropic
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ============================================================================
# STEP 1: SMART CHUNKING - Break text into meaningful sections
# ============================================================================

class SmartChunker:
    """Intelligently chunk healthcare policy documents"""
    
    def __init__(self, chunk_size=1000, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_sections(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Break text into semantic chunks based on sections/topics
        """
        chunks = []
        
        # Method 1: Split by headers (if they exist)
        # Common patterns: "SECTION 5:", "Prior Authorization", etc.
        section_pattern = r'\n([A-Z][A-Z\s]{10,})\n'
        sections = re.split(section_pattern, text)
        
        if len(sections) > 3:  # If we found sections
            current_section = "Introduction"
            for i in range(0, len(sections), 2):
                if i + 1 < len(sections):
                    current_section = sections[i].strip()
                    content = sections[i + 1]
                else:
                    content = sections[i]
                
                # Further split if section is too large
                if len(content) > self.chunk_size:
                    sub_chunks = self._sliding_window_chunk(content)
                    for idx, sub_chunk in enumerate(sub_chunks):
                        chunks.append(self._create_chunk(
                            text=sub_chunk,
                            section=current_section,
                            metadata=metadata,
                            chunk_idx=len(chunks)
                        ))
                else:
                    chunks.append(self._create_chunk(
                        text=content,
                        section=current_section,
                        metadata=metadata,
                        chunk_idx=len(chunks)
                    ))
        else:
            # Method 2: Sliding window if no clear sections
            sub_chunks = self._sliding_window_chunk(text)
            for idx, chunk_text in enumerate(sub_chunks):
                chunks.append(self._create_chunk(
                    text=chunk_text,
                    section="General",
                    metadata=metadata,
                    chunk_idx=idx
                ))
        
        return chunks
    
    def _sliding_window_chunk(self, text: str) -> List[str]:
        """Create overlapping chunks using sliding window"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                sentence_end = text.rfind('.', start, end)
                if sentence_end != -1 and sentence_end > start + self.chunk_size * 0.7:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.overlap
        
        return chunks
    
    def _create_chunk(self, text: str, section: str, metadata: Dict, chunk_idx: int) -> Dict:
        """Create a chunk dictionary with metadata"""
        # Extract potential topics/keywords
        topics = self._extract_topics(text)
        
        return {
            "chunk_id": f"{metadata.get('filename', 'unknown')}_{chunk_idx}",
            "text": text,
            "section": section,
            "topics": topics,
            "payer": metadata.get("payer", "unknown"),
            "source_file": metadata.get("filename", "unknown"),
            "char_count": len(text),
            "metadata": metadata
        }
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text using keyword matching"""
        topics = []
        
        topic_keywords = {
            "prior_authorization": ["prior authorization", "pre-authorization", "preauth"],
            "claims": ["claim", "claims submission", "billing"],
            "eligibility": ["eligibility", "enrollment", "coverage"],
            "medical_records": ["medical record", "documentation", "chart"],
            "deadlines": ["deadline", "timeframe", "within.*days"],
            "coding": ["ICD", "CPT", "diagnosis code", "procedure code"],
            "appeals": ["appeal", "grievance", "dispute"],
            "referrals": ["referral", "specialist"],
        }
        
        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if re.search(keyword, text_lower):
                    topics.append(topic)
                    break
        
        return list(set(topics))  # Remove duplicates


# ============================================================================
# STEP 2: EMBEDDING GENERATION - Convert text to vectors
# ============================================================================

class EmbeddingGenerator:
    """Generate embeddings using sentence transformers"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize embedding model
        all-MiniLM-L6-v2: Fast, good quality (384 dimensions)
        all-mpnet-base-v2: Better quality, slower (768 dimensions)
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully!")
    
    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for all chunks"""
        texts = [chunk['text'] for chunk in chunks]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()
        
        return chunks


# ============================================================================
# STEP 3: VECTOR DATABASE - Store and search embeddings
# ============================================================================

class VectorStore:
    """Manage vector database using ChromaDB"""
    
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize ChromaDB"""
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="healthcare_policies",
            metadata={"description": "Healthcare policy documents with embeddings"}
        )
        print(f"Vector store initialized at: {persist_directory}")
    
    def add_chunks(self, chunks: List[Dict]):
        """Add chunks to vector database"""
        if not chunks:
            return
        
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        ids = [chunk['chunk_id'] for chunk in chunks]
        embeddings = [chunk['embedding'] for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [{
            'section': chunk['section'],
            'payer': chunk['payer'],
            'source_file': chunk['source_file'],
            'topics': ','.join(chunk['topics'])
        } for chunk in chunks]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print("Chunks added successfully!")
    
    def search(self, query: str, n_results=5, filter_dict=None) -> List[Dict]:
        """
        Search for relevant chunks
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filter_dict: Optional filters like {"payer": "ANTHEM"}
        """
        # Generate query embedding
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])[0].tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'chunk_id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
    def get_collection_stats(self):
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection.name
        }


# ============================================================================
# STEP 4: RAG PIPELINE - Put it all together
# ============================================================================

class RAGPipeline:
    """Complete RAG pipeline for processing and querying documents"""
    
    def __init__(self, persist_directory="./chroma_db"):
        self.chunker = SmartChunker(chunk_size=1000, overlap=200)
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore(persist_directory)
    
    def process_json_files(self, json_dir: str):
        """
        Process all JSON files from your existing pipeline
        
        Args:
            json_dir: Directory containing your JSON output files
        """
        json_files = list(Path(json_dir).glob("*.json"))
        print(f"Found {len(json_files)} JSON files to process")
        
        all_chunks = []
        
        for json_file in json_files:
            print(f"\nProcessing: {json_file.name}")
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract text and metadata
            text = data.get('sample_text', '')
            metadata = data.get('metadata', {})
            
            if not text:
                print(f"Skipping {json_file.name} - no text found")
                continue
            
            # Chunk the text
            chunks = self.chunker.chunk_by_sections(text, metadata)
            print(f"  Created {len(chunks)} chunks")
            
            all_chunks.extend(chunks)
        
        # Generate embeddings for all chunks
        print("\n" + "="*60)
        all_chunks = self.embedder.generate_embeddings(all_chunks)
        
        # Add to vector store
        print("="*60)
        self.vector_store.add_chunks(all_chunks)
        
        # Save chunks as JSON for reference
        output_file = Path(json_dir) / "processed_chunks.json"
        with open(output_file, 'w') as f:
            # Remove embeddings for JSON storage (too large)
            chunks_without_embeddings = [
                {k: v for k, v in chunk.items() if k != 'embedding'}
                for chunk in all_chunks
            ]
            json.dump(chunks_without_embeddings, f, indent=2)
        
        print(f"\nProcessed chunks saved to: {output_file}")
        print(f"Total chunks in vector store: {len(all_chunks)}")
        
        return all_chunks
    
    def query(self, question: str, n_results=3, payer_filter=None) -> Dict:
        """
        Query the knowledge base
        
        Args:
            question: User's question
            n_results: Number of relevant chunks to retrieve
            payer_filter: Optional filter like "ANTHEM"
        
        Returns:
            Dictionary with results and context
        """
        filter_dict = {"payer": payer_filter} if payer_filter else None
        
        results = self.vector_store.search(
            query=question,
            n_results=n_results,
            filter_dict=filter_dict
        )
        
        return {
            "question": question,
            "relevant_chunks": results,
            "context": "\n\n".join([r['text'] for r in results])
        }


# ============================================================================
# STEP 5: CLAUDE-POWERED BOT - Answer questions using RAG
# ============================================================================

class HealthcarePolicyBot:
    """Bot that answers questions using RAG + Claude"""
    
    def __init__(self, api_key: str, rag_pipeline: RAGPipeline):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.rag_pipeline = rag_pipeline
    
    def ask(self, question: str, payer_filter=None) -> str:
        """
        Ask a question and get an answer
        
        Args:
            question: User's question
            payer_filter: Optional payer filter like "ANTHEM"
        """
        # Retrieve relevant context
        print(f"\n🔍 Searching knowledge base for: {question}")
        rag_results = self.rag_pipeline.query(
            question,
            n_results=5,
            payer_filter=payer_filter
        )
        
        context = rag_results['context']
        sources = [r['metadata'] for r in rag_results['relevant_chunks']]
        
        # Create prompt for Claude
        prompt = f"""You are a healthcare policy expert assistant. Answer the user's question based on the provided context from healthcare policy documents.

Context from policy documents:
{context}

User question: {question}

Instructions:
- Answer based primarily on the context provided
- Be specific and cite relevant policy details
- If the context doesn't contain enough information, say so
- Include relevant details like timeframes, requirements, or procedures
- Be clear and concise

Answer:"""
        
        # Get response from Claude
        print("🤖 Generating answer with Claude...")
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer = message.content[0].text
        
        # Format response with sources
        response = f"{answer}\n\n📚 Sources:\n"
        for i, source in enumerate(sources[:3], 1):
            response += f"{i}. {source['source_file']} - Section: {source['section']}\n"
        
        return response


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Step 1: Initialize RAG pipeline
    print("Initializing RAG Pipeline...")
    rag = RAGPipeline(persist_directory="./healthcare_vector_db")
    
    # Step 2: Process your existing JSON files
    json_directory = "./final_json_output"  # Change to your directory
    rag.process_json_files(json_directory)
    
    # Step 3: Get stats
    stats = rag.vector_store.get_collection_stats()
    print(f"\n✅ Setup complete! Vector store contains {stats['total_chunks']} chunks")
    
    # Step 4: Test queries (without Claude for now)
    print("\n" + "="*60)
    print("Testing semantic search...")
    print("="*60)
    
    test_queries = [
        "What requires prior authorization?",
        "How do I submit medical records?",
        "What are the claim filing deadlines?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = rag.query(query, n_results=2)
        print(f"Found {len(results['relevant_chunks'])} relevant chunks")
        for i, chunk in enumerate(results['relevant_chunks'][:1], 1):
            print(f"\nChunk {i} (from {chunk['metadata']['source_file']}):")
            print(chunk['text'][:200] + "...")
    
    # Step 5: Use with Claude (uncomment and add your API key)
    # print("\n" + "="*60)
    # print("Testing Claude-powered bot...")
    # print("="*60)
    # 
    # bot = HealthcarePolicyBot(
    #     api_key="your-api-key-here",
    #     rag_pipeline=rag
    # )
    # 
    # answer = bot.ask("What requires prior authorization for Anthem?")
    # print(answer)