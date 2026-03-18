import os
import argparse
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank

from create_chunks import load_chunk_database


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_log = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "jina-reranker-v2-base-multilingual"

def get_vector_store(
    persist_directory: str = "./vector-db",
    hf_api_key: str = None
) -> Chroma:
    """
    Initializes and returns the Chroma Vector Store using HuggingFace embeddings.
    """
    model_name = EMBEDDING_MODEL
    
    hf_api_key = hf_api_key or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not hf_api_key:
        _log.error("Missing Hugging Face API Token (HUGGINGFACEHUB_API_TOKEN)")
        raise ValueError("Missing Hugging Face API Token (HUGGINGFACEHUB_API_TOKEN)")

    embeddings = HuggingFaceEndpointEmbeddings(
        model=model_name,
        huggingfacehub_api_token=hf_api_key,
    )

    vector_store = Chroma(
        collection_name="docling_chunks",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    return vector_store


def build_retriever(
    vector_store: Chroma,
    jina_api_key: str = None,
    initial_k: int = 20,
    final_k: int = 5
):
    """
    Builds a LangChain retriver that first fetches `initial_k` docs from Chroma, 
    and then rerank them down to the `final_k` best results.
    """
    jina_api_key = jina_api_key or os.environ.get("JINA_API_KEY")
    if not jina_api_key:
         _log.error("Missing Jina API Key (JINA_API_KEY)")
         raise ValueError("Missing Jina API Key (JINA_API_KEY)")
         
    base_retriever = vector_store.as_retriever(search_kwargs={"k": initial_k})
    
    compressor = JinaRerank(
        jina_api_key=jina_api_key,
        model=RERANKER_MODEL,
        top_n=final_k
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    return compression_retriever

def main(folder: Path, persist_directory: Path):
    _log.info("Connecting to Vector Store (Chroma)...")
    store = get_vector_store(persist_directory=str(persist_directory))
    
    existing_ids = store.get()["ids"]
    if len(existing_ids) == 0:
        _log.info(f"Chroma is empty. Loading and ingesting documents from {folder}...")
        json_files = list(folder.glob("*.json"))
        if not json_files:
            _log.warning(f"No JSON files found in {folder}")
            return store

        docs = load_chunk_database(json_files)
        if docs:
            _log.info(f"Injecting {len(docs)} documents into Chroma...")
            store.add_documents(docs)
            _log.info("Injection complete!")
        else:
            _log.warning("No documents to inject.")
    else:
        _log.info(f"Chroma already contains {len(existing_ids)} documents. Skipping injection.")

    return store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Vector DB and Test Reranker")
    parser.add_argument("folder", type=Path, default=Path("scratch"), help="Folder containing Docling JSON files.")
    parser.add_argument("--persist_directory", type=Path, default=Path("vector-db"), help="Directory to store the vector database.")
    parser.add_argument("--initial_k", type=int, default=20, help="Number of documents to fetch from Chroma.")
    parser.add_argument("--final_k", type=int, default=5, help="Number of documents to return after reranking.")
    parser.add_argument("--query", type=str, default="What are the essential characteristics of a 4X strategy game?", help="Validation query")
    
    args = parser.parse_args()
    
    try:
        store = main(args.folder, args.persist_directory)
        
        if args.query:
            _log.info(f"Executing Dual-Stage Search for query: '{args.query}'...")
            retriever = build_retriever(vector_store=store, initial_k=args.initial_k, final_k=args.final_k)
            
            final_docs = retriever.invoke(args.query)
        
            _log.info(f"Reranking Complete! Top {len(final_docs)} perfectly aligned chunks:")
            print("-" * 50)
            for i, doc in enumerate(final_docs, 1):
                print(f"Result #{i} | Page {doc.metadata.get('page_number', '?')} | Headings: {doc.metadata.get('headings', [])}")
                print(f"Snippet: {doc.page_content[:150]}...")
                if doc.metadata.get('element_types'):
                    print(f"Elements: {doc.metadata.get('element_types')}")
                print("-" * 50)
    except Exception as e:
        _log.error(f"An error occurred: {e}", exc_info=True)
