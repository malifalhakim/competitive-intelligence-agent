import json
import argparse
from pathlib import Path
from typing import List

from docling_core.types.doc import DoclingDocument
from docling.chunking import HierarchicalChunker
from langchain_core.documents import Document


def load_docling_json_to_langchain(json_path: Path) -> List[Document]:
    """
    Loads a Docling output JSON file and processes it into LangChain Document chunks.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
        
    print(f"Loading document from {json_path}...")
    with json_path.open("r", encoding="utf-8") as f:
        doc_dict = json.load(f)
        
    doc = DoclingDocument.model_validate(doc_dict)
    
    print("Chunking document...")
    chunker = HierarchicalChunker()
    doc_chunks = chunker.chunk(doc)
    
    langchain_docs = []
    for chunk in doc_chunks:
        metadata = {
            "source_filename": doc.name,
            "headings": chunk.meta.headings if chunk.meta.headings else [],
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
                            
            metadata["element_types"] = list(element_types)
            if pictures:
                metadata["pictures"] = pictures
            if tables:
                metadata["tables_html"] = tables

        lc_doc = Document(
            page_content=chunk.text,
            metadata=metadata
        )
        langchain_docs.append(lc_doc)
        
    return langchain_docs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Docling JSON to Langchain Chunks")
    parser.add_argument("json_path", type=Path, help="Path to the JSON file to chunk.")
    parser.add_argument("--output_json", type=Path, help="Optional path to save the resulting LangChain chunks as JSON.")
    
    args = parser.parse_args()
    
    try:
        chunks = load_docling_json_to_langchain(args.json_path)
        if args.output_json:
            print(f"Saving {len(chunks)} chunks to {args.output_json}...")
            chunks_data = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
            with args.output_json.open("w", encoding="utf-8") as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)
            print(f"Successfully saved to {args.output_json}")
        else:
            print(f"\\nSuccessfully created {len(chunks)} LangChain Document chunks!")
            print("-" * 50)
            print("Example of the first chunk:")
            print(repr(chunks[0]))
            print("-" * 50)
    except Exception as e:
        print(f"Error processing {args.json_path}: {e}")
