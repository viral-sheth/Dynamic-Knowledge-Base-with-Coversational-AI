"""
Azure Blob Storage Integration for PDF Crawler
Automatically uploads crawled PDFs to Azure Storage instead of saving locally
"""

import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, ContentSettings
from datetime import datetime
import logging
from typing import Optional, Dict
import hashlib

class AzurePDFUploader:
    """
    Handles uploading PDFs directly to Azure Blob Storage
    """
    
    def __init__(self, connection_string: str, container_name: str = "insurance-pdfs"):
        """
        Initialize Azure Blob Storage connection
        
        Args:
            connection_string: Azure Storage connection string
            container_name: Name of the container to store PDFs
        """
        self.connection_string = connection_string
        self.container_name = container_name
        
        # Set up logging FIRST
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Then connect to Azure
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Create container if it doesn't exist
        self._ensure_container_exists()
    
    def _ensure_container_exists(self):
        """Create container if it doesn't exist"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            if not container_client.exists():
                container_client.create_container()
                self.logger.info(f"Created container: {self.container_name}")
        except Exception as e:
            self.logger.error(f"Error creating container: {e}")
            raise
    
    def upload_pdf_from_url(self, pdf_url: str, pdf_content: bytes, 
                           payer_name: str, metadata: Optional[Dict] = None) -> str:
        """
        Upload PDF directly to Azure Storage
        
        Args:
            pdf_url: Original URL of the PDF
            pdf_content: PDF file content as bytes
            payer_name: Name of the insurance payer (e.g., 'anthem', 'uhc')
            metadata: Optional metadata dictionary
            
        Returns:
            Azure blob URL of uploaded PDF
        """
        try:
            # Generate blob name with organized structure
            blob_name = self._generate_blob_name(pdf_url, payer_name)
            
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            # Prepare metadata
            blob_metadata = {
                "source_url": pdf_url,
                "payer_name": payer_name,
                "upload_date": datetime.utcnow().isoformat(),
                "content_hash": hashlib.md5(pdf_content).hexdigest()
            }
            
            if metadata:
                blob_metadata.update(metadata)
            
            # Upload to Azure
            from azure.storage.blob import ContentSettings
            
            blob_client.upload_blob(
                pdf_content,
                overwrite=True,
                metadata=blob_metadata,
                content_settings=ContentSettings(content_type="application/pdf")
            )
            
            blob_url = blob_client.url
            self.logger.info(f"Successfully uploaded: {blob_name}")
            
            return blob_url
            
        except Exception as e:
            self.logger.error(f"Error uploading PDF from {pdf_url}: {e}")
            raise
    
    def upload_pdf_from_file(self, file_path: str, payer_name: str, 
                            metadata: Optional[Dict] = None) -> str:
        """
        Upload PDF file to Azure Storage
        
        Args:
            file_path: Local path to PDF file
            payer_name: Name of the insurance payer
            metadata: Optional metadata dictionary
            
        Returns:
            Azure blob URL of uploaded PDF
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_content = file.read()
            
            # Use filename as pseudo-URL for blob naming
            pdf_url = os.path.basename(file_path)
            
            return self.upload_pdf_from_url(pdf_url, pdf_content, payer_name, metadata)
            
        except Exception as e:
            self.logger.error(f"Error uploading file {file_path}: {e}")
            raise
    
    def _generate_blob_name(self, pdf_url: str, payer_name: str) -> str:
        """
        Generate organized blob name with folder structure
        
        Structure: payer_name/year-month/filename_timestamp.pdf
        """
        # Extract filename from URL
        filename = pdf_url.split('/')[-1]
        if not filename.endswith('.pdf'):
            filename = f"{filename}.pdf"
        
        # Remove special characters
        filename = "".join(c for c in filename if c.isalnum() or c in '._-')
        
        # Create folder structure
        now = datetime.utcnow()
        folder_path = f"{payer_name}/{now.year}-{now.month:02d}"
        
        # Add timestamp to prevent duplicates
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        base_name = filename.rsplit('.', 1)[0]
        blob_name = f"{folder_path}/{base_name}_{timestamp}.pdf"
        
        return blob_name
    
    def list_pdfs(self, payer_name: Optional[str] = None) -> list:
        """
        List all PDFs in storage, optionally filtered by payer
        
        Args:
            payer_name: Filter by payer name (optional)
            
        Returns:
            List of blob names
        """
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            
            if payer_name:
                prefix = f"{payer_name}/"
                blobs = container_client.list_blobs(name_starts_with=prefix)
            else:
                blobs = container_client.list_blobs()
            
            return [blob.name for blob in blobs]
            
        except Exception as e:
            self.logger.error(f"Error listing PDFs: {e}")
            return []
    
    def delete_pdf(self, blob_name: str) -> bool:
        """
        Delete a PDF from storage
        
        Args:
            blob_name: Name of the blob to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            blob_client.delete_blob()
            self.logger.info(f"Deleted: {blob_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting {blob_name}: {e}")
            return False


# Integration example for your crawler
class EnhancedPayerCrawler:
    """
    Enhanced crawler that uploads PDFs directly to Azure Storage
    """
    
    def __init__(self, azure_connection_string: str, container_name: str = "pdfs"):
        self.azure_uploader = AzurePDFUploader(azure_connection_string, container_name)
        self.logger = logging.getLogger(__name__)
    
    def download_and_upload_pdf(self, pdf_url: str, payer_name: str, 
                                session=None) -> Optional[str]:
        """
        Download PDF and upload directly to Azure Storage
        
        Args:
            pdf_url: URL of the PDF to download
            payer_name: Name of the insurance payer
            session: Optional requests session for download
            
        Returns:
            Azure blob URL if successful, None otherwise
        """
        import requests
        
        try:
            # Download PDF
            if session:
                response = session.get(pdf_url, timeout=30)
            else:
                response = requests.get(pdf_url, timeout=30)
            
            response.raise_for_status()
            pdf_content = response.content
            
            # Prepare metadata
            metadata = {
                "file_size": str(len(pdf_content)),
                "source_domain": pdf_url.split('/')[2] if '://' in pdf_url else 'unknown'
            }
            
            # Upload to Azure
            blob_url = self.azure_uploader.upload_pdf_from_url(
                pdf_url=pdf_url,
                pdf_content=pdf_content,
                payer_name=payer_name,
                metadata=metadata
            )
            
            self.logger.info(f"PDF uploaded to Azure: {blob_url}")
            return blob_url
            
        except Exception as e:
            self.logger.error(f"Error processing {pdf_url}: {e}")
            return None
    
    def batch_upload_pdfs(self, pdf_urls: list, payer_name: str) -> Dict[str, str]:
        """
        Upload multiple PDFs to Azure Storage
        
        Args:
            pdf_urls: List of PDF URLs to download and upload
            payer_name: Name of the insurance payer
            
        Returns:
            Dictionary mapping source URLs to Azure blob URLs
        """
        results = {}
        
        for pdf_url in pdf_urls:
            blob_url = self.download_and_upload_pdf(pdf_url, payer_name)
            if blob_url:
                results[pdf_url] = blob_url
        
        self.logger.info(f"Uploaded {len(results)}/{len(pdf_urls)} PDFs for {payer_name}")
        return results


if __name__ == "__main__":
    # Example usage
    
    # Get connection string from environment variable
    CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    
    if not CONNECTION_STRING:
        print("Please set AZURE_STORAGE_CONNECTION_STRING environment variable")
        exit(1)
    
    # Initialize crawler
    crawler = EnhancedPayerCrawler(CONNECTION_STRING)
    
    # Example: Upload PDFs for Anthem
    example_pdfs = [
        "https://example.com/anthem/policy1.pdf",
        "https://example.com/anthem/policy2.pdf"
    ]
    
    results = crawler.batch_upload_pdfs(example_pdfs, "anthem")
    
    print(f"\nUploaded {len(results)} PDFs:")
    for source, azure_url in results.items():
        print(f"  {source} -> {azure_url}")