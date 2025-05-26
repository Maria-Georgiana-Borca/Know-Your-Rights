from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import re
import fitz
from pathlib import Path
# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def clean_text(text):
    text = re.sub(r'Pagina?\s+\d+\s+din\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+/\d+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\d{2}\.\d{2}\.\d{4}', '', text)
    
    # Normalizează spațiile
    text = re.sub(r'\s{3,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def identify_doc_type(filename):
    filename = filename.lower()
    if 'codul-muncii' in filename or 'cod muncii' in filename:
        return 'CM'
    elif 'constitutia' in filename:
        return 'CONST'
    elif 'cod-civil' in filename or 'cod civil' in filename:
        return 'CC'
    elif 'cod-fiscal' in filename or 'cod fiscal' in filename:
        return 'CF'
    else:
        return 'LEGAL'

def is_article_reference(text_line):

    reference_patterns = [
        r'^\s*(?:conform|potrivit|prevederile|prevazut\s+(?:la|de)|dispozitiile|in\s+sensul|in\s+conditiile)\s+(?:art\.|articolul)\s+\d+',
        r'(?:conform|potrivit|prevederile|prevazut\s+(?:la|de)|dispozitiile)\s+(?:art\.|articolul)\s+\d+(?:\s*[-,.]|\s*$)',
        r'^\s*(?:art\.|articolul)\s+\d+\s*[-,.]?\s*(?:alin|lit|pct)\.',  # art. 132 alin. 1
        r'la\s+(?:art\.|articolul)\s+\d+',  # la art. 132
        r'din\s+(?:art\.|articolul)\s+\d+', # din art. 132
        r'de\s+(?:art\.|articolul)\s+\d+'
    ]

    text_lower = text_line.lower()
    for pattern in reference_patterns:
        if re.search(pattern, text_lower):
            return True
    return False

def extract_articles(text, doc_type):
    print(f"DEBUG pentru {doc_type}:")
    chunks = []
    
    ARTICLE_RE = re.compile(
        r'(?:^|\s)(?:ARTICOLUL|Articolul|ART\.?|Art\.)\s+(\d+(?:\^\d+)?)\.?\b'
        r'(?:\s*[-–]\s*)?(?:\s*\n?\s*(.+?))?(?=\s|$|\n)',
        flags=re.MULTILINE
    )
    
    ARTICLE_REFERENCE = re.compile(
        r'(?:conform|potrivit|prevederile|prevazut\s+(?:la|de)|dispozitiile|la|din)\s+(?:art\.|articolul)\s+\d+\b',
        flags=re.MULTILINE | re.IGNORECASE
    )
    
    all_matches = ARTICLE_RE.findall(text)
    print(f"   Total articole găsite cu ARTICLE_RE: {len(all_matches)}")
    if all_matches:
        print(f"   Primele 10: {all_matches[:10]}")
    
    # Gasește referințele pentru a le exclude ulterior
    references = ARTICLE_REFERENCE.findall(text)
    print(f"   Referințe găsite (de exclus): {len(references)}")
    if references:
        print(f"   Primele referințe: {references[:5]}")
    
    # Iterează prin toate meciurile găsite
    for match in ARTICLE_RE.finditer(text):
        full_match = match.group(0)
        article_num = match.group(1)
        article_title = match.group(2) if match.group(2) else ""
            
        # Verificare mai riguroasă pentru referințe
        context_start = max(0, match.start() - 100)  # Context mai mare
        context_end = min(len(text), match.end() + 50)
        context = text[context_start:context_end]
        
        # Verificăm dacă articolul este într-un context de referință
        is_reference = False
        
        # 1. Verificăm contextul din jurul match-ului
        if ARTICLE_REFERENCE.search(context):
            is_reference = True
            
        # 2. Verificăm dacă linia întreagă pare a fi o referință
        line_start = text.rfind('\n', 0, match.start()) + 1
        line_end = text.find('\n', match.end())
        if line_end == -1:
            line_end = len(text)
        full_line = text[line_start:line_end]
        
        if is_article_reference(full_line):
            is_reference = True
            
        # 3. Verificăm dacă nu urmează conținut substanțial
        content_preview = text[match.end():match.end()+10].strip()
        if len(content_preview) < 10 and re.search(r'(?:art\.|articolul)\s+\d+', content_preview, re.IGNORECASE):
            is_reference = True
        
        if not is_reference:
            # Extragem conținutul articolului
            start_pos = match.end()
            
            # Găsim sfârșitul articolului (următorul articol sau sfârșitul textului)
            next_article = ARTICLE_RE.search(text, start_pos)
            if next_article:
                end_pos = next_article.start()
                content = text[start_pos:end_pos].strip()
            else:
                content = text[start_pos:].strip()
            
            # Curățăm conținutul de linii goale multiple
            content = re.sub(r'\n{3,}', '\n\n', content)
            content = content.strip()
            
            if content and len(content) > 10:  # Doar dacă avem conținut substanțial
                chunk = {
                    'id': f"{doc_type}_art_{article_num}",
                    'text': content,
                    'metadata': {
                        'doc_type': doc_type,
                        'article_number': article_num,
                        'article_title': article_title.strip() if article_title else ""
                    }
                }
                chunks.append(chunk)
            else:
                print(f"Articol {article_num} - conținut prea scurt: '{content[:50]}'")
    
    print(f"   🎯 FINAL: {len(chunks)} articole extrase din {doc_type}")
    return chunks

def process_pdf(pdf_path):
    print(f"Procesez: {pdf_path.name}")
    
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"   Lungime text brut: {len(raw_text)} caractere")
    
    clean_text_content = clean_text(raw_text)
    print(f"   Lungime text curat: {len(clean_text_content)} caractere")
    
    doc_type = identify_doc_type(pdf_path.name)
    print(f"   Tip document: {doc_type}")
    
    chunks = extract_articles(clean_text_content, doc_type)
    print(f"{len(chunks)} articole extrase")
    
    return chunks

def create_vector_store(persist_directory: str) -> Chroma:
    """Create a vector store."""
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = Chroma(
        collection_name="know_your_rights",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vector_store

if __name__ == "__main__":
    vector_db = create_vector_store(persist_directory="/Users/mariaborca/Documents/AI_2023-2024/Semestrul 4/Machine Learning/Know-Your-Rights/chroma_db_articles")

    directory = Path("/Users/mariaborca/Documents/AI_2023-2024/Semestrul 4/Machine Learning/Know-Your-Rights/data")
    pdf_files = list(directory.glob("*.pdf"))
    for pdf in pdf_files:
        print(f"Processing {pdf.name}")
        chunks = process_pdf(pdf)
        documents = [
            Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"]
            )
            for chunk in chunks
        ]
        vector_db.add_documents(documents)