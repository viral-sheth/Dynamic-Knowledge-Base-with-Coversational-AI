"""
Save scraped PDFs directly to Azure PostgreSQL (no LLM extraction)
Stores PDF as binary data in database
"""

import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class PDFDatabaseUploader:
    """Upload PDF files directly to PostgreSQL"""
    
    def __init__(self):
        """Initialize database connection"""
        self.conn_string = (
            f"host={os.getenv('DATABASE_HOST')} "
            f"port={os.getenv('DATABASE_PORT', '5432')} "
            f"dbname={os.getenv('DATABASE_NAME', 'insurance_kb')} "
            f"user={os.getenv('DATABASE_USER')} "
            f"password={os.getenv('DATABASE_PASSWORD')} "
            f"sslmode={os.getenv('DATABASE_SSL_MODE', 'require')}"
        )
        self.conn = None
    
    def connect(self):
        """Connect to database"""
        try:
            self.conn = psycopg2.connect(self.conn_string)
            logger.info("✓ Connected to Azure PostgreSQL")
            self.create_table_if_needed()
        except Exception as e:
            logger.error(f"Failed to connect: {str(e)}")
            raise
    
    def create_table_if_needed(self):
        """Create table for storing PDFs if it doesn't exist"""
        with self.conn.cursor() as cur:
            # Create providers table if needed
            cur.execute("""
                CREATE TABLE IF NOT EXISTS insurance_providers (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create PDF storage table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pdf_documents (
                    id SERIAL PRIMARY KEY,
                    provider_id INTEGER REFERENCES insurance_providers(id),
                    filename VARCHAR(500) NOT NULL,
                    file_path TEXT,
                    file_size_mb DECIMAL(10,2),
                    pdf_url TEXT,
                    pdf_data BYTEA,
                    state_specific VARCHAR(2),
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(provider_id, filename)
                )
            """)
            
            self.conn.commit()
            logger.info("✓ Tables ready")
    
    def get_or_create_provider(self, provider_name: str) -> int:
        """Get or create provider and return ID"""
        with self.conn.cursor() as cur:
            # Check if exists
            cur.execute("SELECT id FROM insurance_providers WHERE name = %s", (provider_name,))
            result = cur.fetchone()
            
            if result:
                return result[0]
            
            # Create new
            cur.execute(
                "INSERT INTO insurance_providers (name) VALUES (%s) RETURNING id",
                (provider_name,)
            )
            provider_id = cur.fetchone()[0]
            self.conn.commit()
            return provider_id
    
    def upload_pdf(self, pdf_path: str, provider_name: str, pdf_url: str = None, state: str = None) -> bool:
        """
        Upload a PDF file to database
        
        Args:
            pdf_path: Path to PDF file
            provider_name: Name of insurance provider
            pdf_url: Original URL of PDF (optional)
            state: State code (e.g., 'OH', 'CA') (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get provider ID
            provider_id = self.get_or_create_provider(provider_name)
            
            # Read PDF file
            with open(pdf_path, 'rb') as f:
                pdf_binary = f.read()
            
            # Get file info
            filename = os.path.basename(pdf_path)
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            
            # Insert into database
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO pdf_documents 
                    (provider_id, filename, file_path, file_size_mb, pdf_url, pdf_data, state_specific)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (provider_id, filename) 
                    DO UPDATE SET 
                        pdf_data = EXCLUDED.pdf_data,
                        file_size_mb = EXCLUDED.file_size_mb,
                        uploaded_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, (provider_id, filename, pdf_path, file_size_mb, pdf_url, pdf_binary, state))
                
                doc_id = cur.fetchone()[0]
                self.conn.commit()
            
            logger.info(f"✓ Uploaded: {filename} (ID: {doc_id}, Size: {file_size_mb:.2f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to upload {pdf_path}: {str(e)}")
            self.conn.rollback()
            return False
    
    def upload_directory(self, pdf_directory: str, provider_name: str):
        """
        Upload all PDFs from a directory
        
        Args:
            pdf_directory: Directory containing PDFs
            provider_name: Name of insurance provider
        """
        # Find all PDFs
        pdf_files = []
        for root, dirs, files in os.walk(pdf_directory):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Found {len(pdf_files)} PDFs in {pdf_directory}")
        logger.info(f"Provider: {provider_name}")
        logger.info(f"{'='*60}\n")
        
        # Upload each PDF
        results = {
            'total': len(pdf_files),
            'successful': 0,
            'failed': 0
        }
        
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"[{i}/{len(pdf_files)}] Uploading: {os.path.basename(pdf_path)}")
            
            # Try to extract state from filename or path
            state = self.extract_state_from_path(pdf_path)
            
            success = self.upload_pdf(pdf_path, provider_name, state=state)
            
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1
        
        logger.info(f"\n{'='*60}")
        logger.info(f"UPLOAD COMPLETE!")
        logger.info(f"{'='*60}")
        logger.info(f"Total PDFs: {results['total']}")
        logger.info(f"✓ Successful: {results['successful']}")
        logger.info(f"✗ Failed: {results['failed']}")
        logger.info(f"{'='*60}\n")
        
        return results
    
    def extract_state_from_path(self, pdf_path: str) -> str:
        """Try to extract state code from filename or path"""
        # Common state codes
        states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
        
        # Check filename and path
        path_upper = pdf_path.upper()
        for state in states:
            if f'{state}_' in path_upper or f'_{state}_' in path_upper or f'/{state}/' in path_upper:
                return state
        
        return None
    
    def list_uploaded_pdfs(self, provider_name: str = None):
        """List all PDFs in database"""
        with self.conn.cursor() as cur:
            if provider_name:
                cur.execute("""
                    SELECT p.name, d.filename, d.file_size_mb, d.state_specific, d.uploaded_at
                    FROM pdf_documents d
                    JOIN insurance_providers p ON d.provider_id = p.id
                    WHERE p.name = %s
                    ORDER BY d.uploaded_at DESC
                """, (provider_name,))
            else:
                cur.execute("""
                    SELECT p.name, d.filename, d.file_size_mb, d.state_specific, d.uploaded_at
                    FROM pdf_documents d
                    JOIN insurance_providers p ON d.provider_id = p.id
                    ORDER BY p.name, d.uploaded_at DESC
                """)
            
            results = cur.fetchall()
            
            print("\n" + "="*60)
            print("PDFs IN AZURE DATABASE")
            print("="*60)
            
            if not results:
                print("\nNo PDFs found in database.")
            else:
                print(f"\nTotal PDFs: {len(results)}\n")
                for provider, filename, size, state, uploaded in results:
                    state_str = f"[{state}]" if state else "[National]"
                    print(f"{provider:20} {state_str:12} {filename:40} {size:6.2f} MB  {uploaded}")
    
    def download_pdf(self, pdf_id: int, output_path: str):
        """Download a PDF from database back to file"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT filename, pdf_data FROM pdf_documents WHERE id = %s", (pdf_id,))
            result = cur.fetchone()
            
            if not result:
                logger.error(f"PDF with ID {pdf_id} not found")
                return False
            
            filename, pdf_data = result
            
            with open(output_path, 'wb') as f:
                f.write(pdf_data)
            
            logger.info(f"✓ Downloaded PDF to: {output_path}")
            return True
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def main():
    """Interactive menu"""
    
    print("\n" + "="*60)
    print("UPLOAD PDFs DIRECTLY TO AZURE DATABASE")
    print("="*60)
    print("\n1. Upload PDFs from directory")
    print("2. List PDFs in database")
    print("3. Download PDF from database")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    uploader = PDFDatabaseUploader()
    uploader.connect()
    
    try:
        if choice == "1":
            # Upload PDFs
            provider = input("\nEnter provider name (e.g., Anthem): ").strip()
            pdf_dir = input(f"Enter PDF directory [default: payer_pdfs/{provider.lower()}]: ").strip()
            
            if not pdf_dir:
                pdf_dir = f"payer_pdfs/{provider.lower()}"
            
            if not os.path.exists(pdf_dir):
                print(f"\n❌ Directory not found: {pdf_dir}")
                return
            
            uploader.upload_directory(pdf_dir, provider)
        
        elif choice == "2":
            # List PDFs
            provider = input("\nEnter provider name (or press Enter for all): ").strip()
            uploader.list_uploaded_pdfs(provider if provider else None)
        
        elif choice == "3":
            # Download PDF
            pdf_id = int(input("\nEnter PDF ID to download: ").strip())
            output_path = input("Enter output path: ").strip()
            uploader.download_pdf(pdf_id, output_path)
        
        elif choice == "4":
            print("Goodbye!")
        
        else:
            print("Invalid choice!")
    
    finally:
        uploader.close()


if __name__ == "__main__":
    # Check configuration
    if not os.getenv("DATABASE_HOST"):
        print("\n❌ Database not configured!")
        print("Make sure your .env file has:")
        print("  DATABASE_HOST=yourserver.postgres.database.azure.com")
        print("  DATABASE_USER=your_username")
        print("  DATABASE_PASSWORD=your_password")
        exit(1)
    
    main()