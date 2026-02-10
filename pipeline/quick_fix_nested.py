#!/usr/bin/env python3
"""
Quick fix: Download PDFs with nested folder support
"""

import os
import shutil
from azure.storage.blob import BlobServiceClient

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass


def download_all_pdfs():
    """Download PDFs handling nested folders"""
    
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    
    if not conn_str:
        print("❌ AZURE_STORAGE_CONNECTION_STRING not set!")
        return False
    
    print("\n" + "="*60)
    print("DOWNLOADING PDFs FROM AZURE (WITH NESTED FOLDER SUPPORT)")
    print("="*60 + "\n")
    
    output_dir = "./temp_pdfs/"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        blob_service = BlobServiceClient.from_connection_string(conn_str)
        container = blob_service.get_container_client("pdfs")
        
        blobs = list(container.list_blobs())
        pdf_blobs = [b for b in blobs if b.name.endswith('.pdf')]
        
        print(f"Found {len(pdf_blobs)} PDFs in Azure\n")
        
        downloaded = []
        failed = []
        
        for i, blob in enumerate(pdf_blobs, 1):
            print(f"[{i}/{len(pdf_blobs)}] {blob.name}")
            
            # Create full local path with folders
            local_path = os.path.join(output_dir, blob.name)
            
            # Create nested directories
            local_dir = os.path.dirname(local_path)
            os.makedirs(local_dir, exist_ok=True)
            
            try:
                # Download the blob
                blob_client = container.get_blob_client(blob.name)
                with open(local_path, 'wb') as f:
                    f.write(blob_client.download_blob().readall())
                
                size_mb = blob.size / (1024*1024)
                print(f"  ✓ Downloaded ({size_mb:.2f} MB)")
                downloaded.append(local_path)
                
            except Exception as e:
                print(f"  ✗ Failed: {str(e)[:60]}")
                failed.append(blob.name)
        
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"  Total PDFs: {len(pdf_blobs)}")
        print(f"  Downloaded: {len(downloaded)}")
        print(f"  Failed: {len(failed)}")
        
        if downloaded:
            print(f"\n  PDFs saved to: {output_dir}")
            
            # Show directory structure
            print("\n  Directory structure:")
            for root, dirs, files in os.walk(output_dir):
                level = root.replace(output_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                folder_name = os.path.basename(root)
                if folder_name:
                    print(f"{indent}📁 {folder_name}/")
                    sub_indent = ' ' * 2 * (level + 1)
                    pdf_files = [f for f in files if f.endswith('.pdf')]
                    if pdf_files:
                        print(f"{sub_indent}  ({len(pdf_files)} PDFs)")
        
        if failed:
            print("\n  ⚠ Failed files:")
            for f in failed[:5]:
                print(f"    - {f}")
            if len(failed) > 5:
                print(f"    ... and {len(failed)-5} more")
        
        print("="*60 + "\n")
        
        return len(downloaded) > 0
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = download_all_pdfs()
    
    if success:
        print("✅ Download complete!")
        print("\nNext step: Run the processing pipeline")
        print("  python azure_pipeline_simple.py")
    else:
        print("❌ Download failed!")