import json
import logging
from pathlib import Path
from typing import List

from docling_core.types.doc import DoclingDocument
from docling.chunking import HierarchicalChunker
from langchain_core.documents import Document


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_log = logging.getLogger(__name__)


def _process_chunk(chunk, doc_name: str) -> Document:
    """
    Processes an individual Docling chunk and converts it into a LangChain Document.
    """
    metadata = {
        "source_filename": doc_name,
        "headings": " > ".join(chunk.meta.headings) if chunk.meta.headings else "",
    }
    
    # --- Add media properties ---
    if hasattr(chunk.meta, 'doc_items') and len(chunk.meta.doc_items) > 0:
        if hasattr(chunk.meta.doc_items[0], 'prov') and len(chunk.meta.doc_items[0].prov) > 0:
            metadata["page_number"] = chunk.meta.doc_items[0].prov[0].page_no

        element_types = set()
        pictures = []
        tables = []
        
        for item in chunk.meta.doc_items:
            lbl = getattr(item, 'label', 'unknown')
            element_types.add(lbl)
            
            if lbl == "picture" or hasattr(item, 'image'):
                pic_data = {}
                
                if hasattr(item, 'image') and item.image and hasattr(item.image, 'uri'):
                    pic_data["uri"] = str(item.image.uri)
                    
                if hasattr(item, 'meta') and hasattr(item.meta, 'classification') and item.meta.classification:
                    if item.meta.classification.predictions:
                        pic_data["classification"] = item.meta.classification.predictions[0].class_name
                        
                if hasattr(item, 'meta') and hasattr(item.meta, 'description') and item.meta.description:
                     pic_data["description"] = item.meta.description.text
                     
                if hasattr(item, 'captions') and item.captions:
                    pic_data["caption"] = item.captions[0].text if hasattr(item.captions[0], 'text') else str(item.captions[0])
                    
                if pic_data:
                    pictures.append(pic_data)
                    
            elif lbl == "table" or hasattr(item, 'export_to_html'):
                if hasattr(item, 'export_to_html'):
                    try:
                        tables.append(item.export_to_html())
                    except Exception:
                        pass
                        
        metadata["element_types"] = ", ".join(element_types) if element_types else ""
        if pictures:
            metadata["pictures"] = json.dumps(pictures)
        if tables:
            metadata["tables_html"] = json.dumps(tables)

    return Document(
        page_content=chunk.text,
        metadata=metadata
    )


def load_docling_json_to_langchain(json_path: Path) -> List[Document]:
    """
    Loads a Docling output JSON file and processes it into LangChain Document chunks.
    """
    if not json_path.exists():
        _log.error(f"JSON file not found: {json_path}")
        return []
        
    _log.info(f"Loading document from {json_path}...")
    try:
        with json_path.open("r", encoding="utf-8") as f:
            doc_dict = json.load(f)
        doc = DoclingDocument.model_validate(doc_dict)
    except Exception as e:
        _log.error(f"Failed to load or validate document {json_path}: {e}", exc_info=True)
        return []
    
    _log.info("Chunking document...")
    try:
        chunker = HierarchicalChunker()
        doc_chunks = chunker.chunk(doc)
    except Exception as e:
        _log.error(f"Failed to chunk document {json_path}: {e}", exc_info=True)
        return []
    
    langchain_docs = []
    for chunk in doc_chunks:
        langchain_docs.append(_process_chunk(chunk, doc.name))
        
    return langchain_docs


def load_chunk_database(json_paths: List[Path]) -> List[Document]:
    """
    Loads LangChain Documents from a list of previously saved JSON files.
    """
    documents = []
    for json_path in json_paths:
        try:
            chunked_docs = load_docling_json_to_langchain(json_path)
            if chunked_docs:
                documents.extend(chunked_docs)
                _log.info(f"Successfully processed {json_path} and extracted {len(chunked_docs)} chunks.")
            else:
                _log.warning(f"No chunks extracted from {json_path}")
        except Exception as e:
            _log.error(f"Unexpected error processing {json_path}: {e}", exc_info=True)
            
    return documents
