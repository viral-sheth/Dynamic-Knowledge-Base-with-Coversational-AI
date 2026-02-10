"""
Configuration for Azure and deduplication system
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Azure Storage Configuration
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "healthcare-policies"

# Policy Settings
MIN_CONFIDENCE_THRESHOLD = 0.7
SIMILARITY_THRESHOLD = 0.85
DEFAULT_POLICY_TTL_DAYS = 730  # 2 years

# Anthropic API (optional, for LLM-based extraction)
#ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", None)

# Directories
TEMP_PDF_DIR = "./temp_pdfs"
TEMP_JSON_DIR = "./temp_json"
OUTPUT_DIR = "./output"