from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def read_pdf(file_path: str) -> Document:
    """Read a PDF file and return its text content."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    
    return Document(
        page_content=text, 
        metadata={"file": os.path.basename(file_path)}
        )

def chunk_document(doc: Document, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    documents = splitter.split_documents([doc])
    for i, chunk in enumerate(documents):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["preview"] = chunk.page_content[:100].strip()
        chunk.metadata["file"] = os.path.basename(file_path)

    return documents

def create_vector_store(persist_directory: str) -> Chroma:
    """Create a vector store."""
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = Chroma(
        collection_name="know_your_rights",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vector_store

def process_pdf(file_path: str, persist_directory: str, vector_store, batch_size = 500) -> None:
    """Process a PDF file and store its text in a vector store."""
    # Read the PDF file
    doc = read_pdf(file_path)
    
    # Split the text into chunks
    documents = chunk_document(doc, file_path)
    print(f"Chunked {len(documents)} chunks from {file_path}.")
    
    # Add documents to the vector store
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        vector_store.add_documents(batch)
    
    print(f"Processed {len(documents)} chunks from {file_path} and stored in {persist_directory}.")

def process_directory(dir_path: str, persist_directory: str, vector_store) -> None:
    for file in os.listdir(dir_path):
        if file.lower().endswith(".pdf"):
            process_pdf(os.path.join(dir_path, file), persist_directory, vector_store)


if __name__ == "__main__":
    # Example usage
    directory_path = "/data"
    persist_directory = "/chromaDB"
    
    vector_store = create_vector_store(persist_directory)
    
    process_directory(directory_path, persist_directory, vector_store)
    
    print("All PDFs processed and stored in the vector store.")