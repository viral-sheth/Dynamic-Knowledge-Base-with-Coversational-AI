from pipeline.azure_pdf_uploader import AzurePDFUploader
from dotenv import load_dotenv
import os
import glob

load_dotenv()

uploader = AzurePDFUploader(
    os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    "insurance-pdfs"
)

# Change this to your local PDF folder
pdf_folder = "./downloads"  # UPDATE THIS PATH!

print("Searching for PDFs...")

# Find all PDFs
pdf_files = glob.glob(f"{pdf_folder}/**/*.pdf", recursive=True)

print(f"Found {len(pdf_files)} PDFs\n")

# Upload each one
for i, pdf_file in enumerate(pdf_files, 1):
    # Get payer name from folder structure
    parts = pdf_file.split(os.sep)
    payer_name = parts[1] if len(parts) > 2 else "unknown"
    
    print(f"[{i}/{len(pdf_files)}] Uploading: {os.path.basename(pdf_file)}")
    
    try:
        blob_url = uploader.upload_pdf_from_file(pdf_file, payer_name)
        print(f"  ✅ Success!")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "="*60)
print("Migration complete!")
print("="*60)
