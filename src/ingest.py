import os
import re
import chromadb
from rank_bm25 import BM25Okapi

# directory where raw markdown docs live
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")


def load_markdown_chunks(docs_dir=DOCS_DIR):
    """
    Splits markdown files by headings (#, ##) so that code blocks
    and their surrounding explanations stay together.
    """
    chunks = []
    
    for filename in os.listdir(docs_dir):
        if not filename.endswith(".md"):
            continue
            
        filepath = os.path.join(docs_dir, filename)
        domain = filename.replace(".md", "")
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # extract base source url if present
        source_match = re.search(r"Source:\s*(https?://[^\s]+)", content)
        source_url = source_match.group(1) if source_match else f"https://docs.{domain}.org"

        # split on markdown headings
        sections = re.split(r"\n(?=##?\s+)", content)
        for idx, sec in enumerate(sections):
            text = sec.strip()
            if not text or len(text) < 30:
                continue
                
            # get the title of this section
            first_line = text.split("\n")[0].replace("#", "").strip()
            chunk_id = f"{domain}_{idx}"
            
            chunks.append({
                "id": chunk_id,
                "text": text,
                "domain": domain,
                "title": first_line,
                "url": f"{source_url}#{first_line.lower().replace(' ', '-')}"
            })

    return chunks


def build_vector_store(chunks, db_dir=CHROMA_DIR):
    """Stores text chunks with metadata in local ChromaDB."""
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(
        name="docuquery_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    
    # check if already populated
    if collection.count() == len(chunks):
        return collection
        
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [{"domain": c["domain"], "title": c["title"], "url": c["url"]} for c in chunks]

    # add to chroma in batch
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas
    )
    return collection


def build_bm25_index(chunks):
    """Builds an in-memory BM25 index for sparse keyword matching."""
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunks


if __name__ == "__main__":
    print("Ingesting markdown documentation...")
    raw_chunks = load_markdown_chunks()
    print(f"Loaded {len(raw_chunks)} semantic chunks across {len(set(c['domain'] for c in raw_chunks))} domains.")
    
    col = build_vector_store(raw_chunks)
    print(f"Indexed {col.count()} documents into ChromaDB at {CHROMA_DIR}.")
