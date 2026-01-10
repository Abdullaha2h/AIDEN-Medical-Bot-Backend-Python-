import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_BACKUP_NAME = os.getenv("PINECONE_INDEX_BACKUP_NAME", "medical-knowledge-backup")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env")

def ingest_backup():
    print("🚀 Starting Backup Index Ingestion (HuggingFace 384d)...")

    # 1. Load Data
    print("📂 Loading PDFs...")
    loader = DirectoryLoader("Medical pdfs/", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"   Loaded {len(documents)} document pages.")

    # 2. Split Text
    print("✂️ Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(text_chunks)} text chunks.")

    # 3. Initialize Embeddings (HuggingFace Only)
    print("🧠 Initializing HuggingFace Embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Test dimension
    test_emb = embeddings.embed_query("test")
    print(f"   Embedding dimension: {len(test_emb)} (Should be 384)")

    # 4. Connect to Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    existing_indexes = [i.name for i in pc.list_indexes()]
    print(f"   Existing indexes: {existing_indexes}")

    if PINECONE_INDEX_BACKUP_NAME not in existing_indexes:
        print(f"⚠️ Index '{PINECONE_INDEX_BACKUP_NAME}' does not exist. Attempting to create it...")
        try:
            from pinecone import ServerlessSpec
            pc.create_index(
                name=PINECONE_INDEX_BACKUP_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"✅ Created index: {PINECONE_INDEX_BACKUP_NAME}")
            # Wait for index to be ready
            while not pc.describe_index(PINECONE_INDEX_BACKUP_NAME).status['ready']:
                time.sleep(1)
            print("   Index is ready!")
        except Exception as e:
            print(f"❌ Failed to create index: {e}")
            print("   Please create it manually in the Pinecone Console: Name='medical-knowledge-backup', Dim=384, Metric='cosine'")
            return

    index = pc.Index(PINECONE_INDEX_BACKUP_NAME)
    print(f"✅ Connected to index: {PINECONE_INDEX_BACKUP_NAME}")

    # 5. Upsert Data
    print("network Uploading vectors to Pinecone...")
    from langchain_community.vectorstores import Pinecone as PineconeStore

    docsearch = PineconeStore.from_documents(
        documents=text_chunks,
        index_name=PINECONE_INDEX_BACKUP_NAME,
        embedding=embeddings
    )
    
    print("🎉 BACKUP INGESTION COMPLETE!")

if __name__ == "__main__":
    ingest_backup()
