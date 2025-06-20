import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(file_path):
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_text(text)

def parse_and_chunk_pdfs(pdf_folder="data"):
    all_chunks = []
    
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            print(f"Processing {filename}")
            path = os.path.join(pdf_folder, filename)
            raw_text = extract_text_from_pdf(path)
            chunks = chunk_text(raw_text)
            all_chunks.extend(chunks)
    
    print(f"\n✅ Parsed {len(all_chunks)} total chunks.")
    return all_chunks

if __name__ == "__main__":
    chunks = parse_and_chunk_pdfs()
    # For inspection:
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---\n{chunk[:500]}...")

