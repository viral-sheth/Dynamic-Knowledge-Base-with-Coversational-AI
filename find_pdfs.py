"""
Find all PDFs in your project directory
"""

import os
import json

def find_all_pdfs(base_dir="."):
    """Find all PDF files in the project"""
    
    pdf_files = []
    
    for root, dirs, files in os.walk(base_dir):
        # Skip hidden directories and common exclude dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', 'env', '__pycache__', 'node_modules']]
        
        for file in files:
            if file.endswith('.pdf'):
                full_path = os.path.join(root, file)
                file_size = os.path.getsize(full_path)
                pdf_files.append({
                    'path': full_path,
                    'name': file,
                    'directory': root,
                    'size_mb': round(file_size / (1024 * 1024), 2)
                })
    
    return pdf_files


def display_pdfs():
    """Display all found PDFs"""
    
    print("="*60)
    print("Searching for PDFs in your project...")
    print("="*60)
    
    pdfs = find_all_pdfs()
    
    if not pdfs:
        print("\n❌ No PDF files found in the current directory!")
        print("\nPossible reasons:")
        print("1. PDFs haven't been downloaded yet")
        print("2. PDFs are in a different directory")
        print("3. You need to run your crawler first")
        print("\nNext steps:")
        print("- Run your crawler to download PDFs first:")
        print("  python payer_portal_crawler.py")
        print("- Or check if PDFs are in a different location")
        return []
    
    print(f"\n✓ Found {len(pdfs)} PDF files:\n")
    
    for i, pdf in enumerate(pdfs, 1):
        print(f"{i}. {pdf['name']}")
        print(f"   Path: {pdf['path']}")
        print(f"   Size: {pdf['size_mb']} MB")
        print()
    
    # Save to JSON for easy reference
    with open('found_pdfs.json', 'w') as f:
        json.dump(pdfs, f, indent=2)
    
    print("="*60)
    print(f"✓ PDF list saved to: found_pdfs.json")
    print("="*60)
    
    return pdfs


if __name__ == "__main__":
    pdfs = display_pdfs()
    
    if pdfs:
        print("\n📝 To test extraction, update test_llm_extraction.py:")
        print(f'   test_pdf = "{pdfs[0]["path"]}"')