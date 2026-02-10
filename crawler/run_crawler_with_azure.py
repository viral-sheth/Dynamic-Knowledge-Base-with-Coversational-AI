"""
Wrapper to run targeted crawler with Azure integration
"""
import sys
import os

from pipeline.azure_integration import PDFToStructuredPipeline
from config import AZURE_CONNECTION_STRING
from crawler.targeted_healthcare_crawler import TargetedHealthcareRuleCrawler

def download_pdfs_from_azure(blob_service, container_name, local_dir="./temp_pdfs"):
    """Download PDFs from Azure to process them locally"""
    os.makedirs(local_dir, exist_ok=True)
    
    container_client = blob_service.get_container_client(container_name)
    pdf_paths = []
    
    print(f"\nDownloading PDFs from Azure container: {container_name}")
    
    for blob in container_client.list_blobs():
        if blob.name.endswith('.pdf'):
            local_path = os.path.join(local_dir, os.path.basename(blob.name))
            
            # Download blob
            blob_client = container_client.get_blob_client(blob.name)
            with open(local_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            
            pdf_paths.append(local_path)
            print(f"  ✓ Downloaded: {os.path.basename(blob.name)}")
    
    return pdf_paths

def main(payer_name="anthem"):
    """Run crawler and process with deduplication"""
    
    print(f"Starting crawler for: {payer_name}")
    
    crawler = TargetedHealthcareRuleCrawler(headless=True, max_depth=2)
    
    try:
        # Step 1: Run targeted crawler (uploads to Azure via AzurePDFUploader)
        crawl_results = crawler.crawl_targeted_companies([payer_name])
    finally:
        crawler.close()
    
    print(f"\n{'='*60}")
    print("PROCESSING WITH DEDUPLICATION")
    print(f"{'='*60}")
    
    # Step 2: Download PDFs from Azure to process them
    pipeline = PDFToStructuredPipeline(AZURE_CONNECTION_STRING)
    
    # Get PDFs from the "pdfs" container (where your crawler uploads)
    pdf_paths = download_pdfs_from_azure(
        pipeline.blob_service,
        container_name="pdfs",  # Your crawler's container
        local_dir="./temp_pdfs"
    )
    
    if pdf_paths:
        print(f"\nProcessing {len(pdf_paths)} PDFs...")
        
        # Step 3: Process with deduplication
        stats = pipeline.process_pdf_batch(
            pdf_paths=pdf_paths,
            payer_name=payer_name
        )
        
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"PDFs processed: {stats['successful']}/{stats['total_pdfs']}")
        print(f"Policies extracted: {stats['total_policies']}")
        print(f"Active policies: {stats['cleanup']['current_policies']}")
        print(f"Archived (expired): {stats['cleanup']['archived_policies']}")
        
        # Cleanup temp files
        print("\nCleaning up temp files...")
        for pdf_path in pdf_paths:
            try:
                os.remove(pdf_path)
            except:
                pass
        
        return stats
    else:
        print("No PDFs found in Azure")
        return None

if __name__ == "__main__":
    payer = sys.argv[1] if len(sys.argv) > 1 else "anthem"
    main(payer)
