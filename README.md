#  Healthcare Payer Knowledge Base & RAG Chatbot

End‑to‑end system used for this submission:

- Targeted crawler that finds payer policy PDFs on provider portals  
- PDFs are uploaded into Azure Blob Storage  
- PDFs are brought back down and converted into structured JSON policies  
- The JSON policies are written back to Azure and deduplicated  
- A combined JSON rules file (for example `healthcare_rules_export.json`) is exported  
- A RAG chatbot (FastAPI backend + React frontend) answers questions on top of that rules file

The rest of this README (below) contains the original crawler‑focused documentation and additional details.

#  Healthcare Payer Knowledge Base

**Automated Healthcare Payer Rule Extraction System**

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org) [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE) [![Selenium](https://img.shields.io/badge/Selenium-4.15-orange)](https://selenium.dev)

> Intelligent web crawler that automatically extracts payer rules, filing requirements, and policies from major healthcare insurance portals, converting unstructured information into structured knowledge for revenue cycle teams.

---

##  **Project Overview**

### **Problem Statement**
Healthcare revenue cycle teams face significant challenges:
- **Manual Portal Navigation**: Staff spend hours searching multiple payer websites
- **Fragmented Information**: Rules scattered across PDFs, portals, and documents  
- **Frequent Policy Changes**: Updates occur regularly without centralized notifications
- **Operational Inefficiency**: Manual processes lead to claim denials and revenue loss
- **Compliance Risks**: Outdated information causes regulatory and financial issues

### **Solution**
Our automated payer portal crawler:
-  **Extracts** payer rules from major healthcare portals automatically
-  **Structures** unorganized data into queryable JSON format
-  **Monitors** policy changes systematically 
-  **Centralizes** knowledge for conversational AI access
-  **Reduces** manual effort by 80%+ for revenue cycle teams

---

##  **System Architecture**

```
┌─────────────────────────────────────────────────────────┐
│                Healthcare Knowledge Base                │
├─────────────────────────────────────────────────────────┤
│   Web Crawler (Selenium)                            │
│     ├── Dynamic content handling                       │
│     ├── Multi-payer portal navigation                  │
│     └── Respectful crawling with rate limits           │
│                                                         │
│   PDF Processor (PyMuPDF + PyPDF2)                  │
│     ├── Dual extraction methods                        │
│     ├── Fallback processing                            │
│     └── Content validation                             │
│                                                         │
│   Rule Extraction Engine                            │
│     ├── Regex pattern matching                         │
│     ├── Content classification                         │
│     ├── Geographic zone detection                      │
│     └── JSON structure generation                      │
│                                                         │
│   Knowledge Base                                     │
│     ├── Structured JSON output                         │
│     ├── Queryable format                               │
│     └── API-ready data                                 │
└─────────────────────────────────────────────────────────┘
```

---

##  **Crawler Versions & Capabilities**

### **1. Basic Crawler** (`crawler/payer_portal_crawler.py`)
**Purpose**: Reliable baseline functionality for single payer extraction

**Features**:
- Direct PDF discovery from provider portals
- Rule extraction with healthcare-specific patterns
- JSON output with structured data
- Rate limiting and respectful crawling

**Proven Results**:
-  **8 PDFs** downloaded from Anthem
-  **880+ pages** of content processed
-  **723 healthcare rules** extracted
-  Categories: Prior Authorization, Timely Filing, Appeals, Claims

**Usage**:
```python
from crawler.payer_portal_crawler import PayerPortalCrawler

crawler = PayerPortalCrawler()
results = crawler.crawl_payer("anthem")
```

### **2. Targeted Healthcare Rule Crawler** (`crawler/targeted_healthcare_crawler.py`)
**Purpose**: Focused discovery of PDFs that contain specific rule types (timely filing, prior auth, billing nuances, appeals) for multiple payers.

**Features**:
- Uses curated payer configurations and Selenium to navigate provider portals  
- Classifies links and PDFs by rule type before downloading  
- Organizes downloads under `./targeted_pdfs/<payer>/<rule_type>/`  
- Optionally uploads PDFs and extracted policies to Azure via `pipeline.azure_pdf_uploader` and `pipeline.policy_deduplication_system`

### **3. Azure-Integrated Crawler Pipeline** (`crawler/run_crawler_with_azure.py` + `pipeline/*`)
**Purpose**: End‑to‑end flow from crawler → Azure Blob Storage → structured policies.

**Features**:
- Runs the targeted crawler for a given payer  
- Downloads PDFs from the Azure `pdfs` container  
- Converts PDFs to policy JSON, deduplicates versions, and stores current policies in Azure  
- Produces a clean policy store that can be exported into `healthcare_rules_export.json` for the RAG chatbot

---

##  **Quick Start Guide**

### **Installation**
```bash
git clone https://github.com/Rithika-vennamaneni/Salud_KnowledgeBase
cd Salud_KnowledgeBase
pip install -r requirements.txt
```

### **Run a basic crawl**
```bash
python crawler/payer_portal_crawler.py
```

### **Run the targeted multi-payer crawler**
```bash
python crawler/targeted_healthcare_crawler.py
```

### **Run the RAG chatbot**
```bash
# 1) Start the FastAPI backend (expects /api/chat)
uvicorn rag.api_server:app --reload --port 8000

# 2a) Serve the static frontend with Python (no build needed)
cd frontend && python -m http.server 5173
# or
# 2b) Serve via npm (if you prefer)
cd frontend && npm install && npm run start

# 3) Open http://localhost:5173
# Set API base to http://localhost:8000 in the UI and click Save.
# Ask questions; include payer names (e.g., "UnitedHealthcare") to narrow results.

Notes:
- The frontend uses CDN React and Babel (type=\"text/babel\") so it runs without a build step. Keep serving from /frontend so ./main.js and ./style.css resolve.
```

---

##  **Configuration**

### **Payer Database** (`payer_companies.csv`)
The system includes 15 major US healthcare payers:

| Company | Type | States | Market Share |
|---------|------|--------|--------------|
| United Healthcare | National | All 50 | 23.0% |
| Anthem/Elevance | Multi-State | 14 states | 8.2% |
| Aetna/CVS Health | National | All 50 | 7.8% |
| Kaiser Permanente | Regional | 9 regions | 5.1% |
| Centene Corporation | Multi-State | 26+ states | 4.8% |

### **Adding New Payers**
Simply update `payer_companies.csv`:
```csv
company_name,base_domain,known_provider_portal,priority,market_share
"New Payer","newpayer.com","https://providers.newpayer.com","high","2.5%"
```

---

##  **Performance Metrics**

### **Discovery Capacity**
- **Basic Crawler**: 8-20 PDFs per payer
- **CSV Crawler**: 50+ provider portals discovered
- **BFS Crawler**: 100+ PDFs per major payer
- **Combined System**: 1,000-3,000 PDFs potential

### **Quality Filtering**
- **78% noise reduction** (removes privacy policies, marketing)
- **167% quality improvement** (relevant healthcare content)
- **Perfect validity rate** (no broken/corrupted files)
- **12+ healthcare terms** per accepted PDF

### **Regional Coverage**
- **Current**: 8/50 US states (16% coverage)
- **Enhanced Potential**: 42-47/50 states (85-95% coverage)
- **Implementation**: 20-30 hours for complete US coverage

---

##  **Advanced Features**

### **Quality Filtering** (`intelligent_pdf_filter.py`)
```python
from pipeline.intelligent_pdf_filter import IntelligentPDFFilter

filter_system = IntelligentPDFFilter()
results = filter_system.process_pdf_batch_with_filtering(pdf_urls)
```

### **Content Analysis** (`pdf_quality_analyzer.py`)
```python
from pipeline.pdf_quality_analyzer import PDFQualityAnalyzer

analyzer = PDFQualityAnalyzer()
results = analyzer.analyze_pdf_batch(pdf_urls)
```

### **Regional Coverage** (`regional_coverage_analyzer.py`)
```python
from pipeline.regional_coverage_analyzer import RegionalCoverageAnalyzer

analyzer = RegionalCoverageAnalyzer()
analysis = analyzer.analyze_payer_regional_coverage(payer_name, pdf_list)
```

---

##  **Production Deployment**

### **Recommended Architecture**
```bash
# 1. Basic extraction for reliable baseline
python crawler/payer_portal_crawler.py

# 2. Targeted multi-payer crawl
python crawler/targeted_healthcare_crawler.py

# 3. Run crawler + Azure pipeline with deduplication (example payer)
python crawler/run_crawler_with_azure.py anthem

# 4. Run policy cleanup/deduplication on Azure store
python -m pipeline.cleanup_policies
```

### **Expected Results**
- **1,500-3,000 high-quality PDFs** discovered
- **85-95% US state coverage** achieved
- **Comprehensive rule extraction** across all major payers
- **Production-ready structured data**

---

##  **Use Cases**

### **Revenue Cycle Management**
- Automated payer rule discovery
- Real-time policy change monitoring
- Compliance verification
- Claim denial reduction

### **Healthcare Operations**
- Prior authorization automation
- Timely filing requirement tracking
- Appeals process optimization
- Provider network management

### **AI/ML Applications**
- Training data for healthcare AI models
- Knowledge base for conversational AI
- Automated rule interpretation
- Predictive compliance analytics

---

##  **AI Assistance Disclosure**

This project was developed with the help of AI coding tools, including:

- GitHub Copilot  
- OpenAI Codex / ChatGPT (via the Codex CLI)  
- Anthropic Claude

These tools were used for code generation, refactoring, and documentation drafts. All AI‑generated content was reviewed, edited, and integrated by the author, who remains responsible for the final design and correctness of the system.

---

##  **References**

- Lewis, Patrick et al. “Retrieval-Augmented Generation for Knowledge-Intensive NLP.” NeurIPS 2020.  
- Sentence-Transformers documentation – https://www.sbert.net/  
- FastAPI documentation – https://fastapi.tiangolo.com/  
- Azure Blob Storage documentation – https://learn.microsoft.com/azure/storage/blobs/  
- Azure Cognitive Services (Document Intelligence) – https://learn.microsoft.com/azure/ai-services/document-intelligence/  
- Example healthcare payer manuals and CMS Medicare Learning Network materials used as representative policy sources (for PDFs converted into this knowledge base).

---

##  **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Healthcare Payer Knowledge Base - Automated Rule Extraction System**
