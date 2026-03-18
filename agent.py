import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Set

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

from vector_store import get_vector_store, build_retriever


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_log = logging.getLogger(__name__)

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

class CompetitorInfo(BaseModel):
    name: str = Field(description="The name of the competitor company or product.")
    features: List[str] = Field(description="A list of specific gameplay features, mechanics, or capabilities. If none found, return an empty array [].")
    price: Optional[float] = Field(description="The numeric price of the game. If it is free to play, output 0.0. If unknown, output null.")
    strengths: List[str] = Field(description="Key strengths, advantages, or unique selling points. If none found, return an empty array []. Do NOT return a string.")
    weaknesses: List[str] = Field(description="Key weaknesses, flaws, or limitations. If none found, return an empty array []. Do NOT return a string like 'None'.")

class InitialDiscoveryList(BaseModel):
    competitors: List[str] = Field(description="A comprehensive list of all competitor names found in the context.")


def get_llm():
    """Initializes the model via Groq API."""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        _log.error("Missing Groq API Key (GROQ_API_KEY)")
        raise ValueError("Missing Groq API Key (GROQ_API_KEY)")
        
    llm = ChatGroq(
        model=MODEL_NAME,
        api_key=groq_api_key,
    )

    return llm


def format_context_for_multimodal(docs: List) -> List[dict]:
    """
    Takes LangChain Documents and converts them into the precise multi-modal 
    message format.
    """
    content_array = []
    
    for i, doc in enumerate(docs):
        text_payload = f"--- Document Chunk {i+1} ---\n{doc.page_content}\n"
        
        if "tables_html" in doc.metadata and doc.metadata["tables_html"]:
            try:
                tables = json.loads(doc.metadata["tables_html"])
                for table in tables:
                    text_payload += f"Found Table:\n{table}\n"
            except json.JSONDecodeError:
                pass
                
        content_array.append({"type": "text", "text": text_payload})
        
        if "pictures" in doc.metadata and doc.metadata["pictures"]:
            try:
                pictures = json.loads(doc.metadata["pictures"])
                for pic in pictures:
                    if "uri" in pic and pic["uri"].startswith("data:image"):
                        content_array.append({
                            "type": "image_url",
                            "image_url": {"url": pic["uri"]}
                        })
            except json.JSONDecodeError:
                pass
                
    return content_array


def discover_competitors(store, llm) -> List[str]:
    """Discover all game titles mentioned in the vector database."""
    _log.info("Executing: Competitor Discovery...")
    discovery_retriever = build_retriever(vector_store=store, initial_k=50, final_k=50)
    
    discovery_query = "game titles, game names, competitor games, mobile games, games"
    discovery_docs = discovery_retriever.invoke(discovery_query)
    
    discovery_parser = PydanticOutputParser(pydantic_object=InitialDiscoveryList)
    all_games: Set[str] = set()
    
    for i, doc in enumerate(discovery_docs):
        _log.info(f"  Scanning chunk {i+1}/{len(discovery_docs)}...")
        
        chunk_content = format_context_for_multimodal([doc])
        chunk_content.append({
            "type": "text",
            "text": (
                "\n\nExtract ALL specific game titles (mobile, PC, or console) mentioned in the text above. "
                "Only include actual game names, NOT company/publisher names. "
                "If no game titles are found, return an empty list.\n\n"
                f"{discovery_parser.get_format_instructions()}"
            )
        })
        
        messages = [
            SystemMessage(content="You are a game title extractor. Return ONLY game names found in the text. If none exist, return an empty competitors list."),
            HumanMessage(content=chunk_content)
        ]
        
        try:
            response = llm.invoke(messages)
            chunk_games = discovery_parser.parse(response.content).competitors
            if chunk_games:
                _log.info(f"    Found in chunk {i+1}: {chunk_games}")
                all_games.update(chunk_games)
        except Exception as e:
            _log.warning(f"    Failed to parse discovery chunk {i+1}: {e}")
    
    competitor_list = sorted(list(all_games))
    _log.info(f"All Competitors Discovered: {competitor_list} ({len(competitor_list)} total)")
    return competitor_list


def analyze_competitor_attributes(comp: str, retriever, llm) -> Dict[str, str]:
    """Gather raw data for a specific competitor's attributes."""
    _log.info(f"  -> Retrieving & Analyzing attributes for: {comp}")
    gathered_data = {}
    
    queries = {
        "features": f"What specific gameplay features or mechanics does the game '{comp}' have?",
        "price": f"How much does the game '{comp}' cost to play or purchase? Looking for a specific number.",
        "strengths": f"What are the main advantages or standout strengths of the game '{comp}' compared to others?",
        "weaknesses": f"What are the known flaws, weaknesses, or limitations of the game '{comp}'?"
    }
    
    for attr, query in queries.items():
        _log.info(f"    - Analyzing {attr.upper()}...")
        attr_docs = retriever.invoke(query)
        
        attr_content = format_context_for_multimodal(attr_docs)
        attr_content.append({"type": "text", "text": f"\n\nBased ONLY on the visual and text context above, comprehensively answer the following query: '{query}'. If the information is not present, state 'No information available.'"})
        
        attr_messages = [
            SystemMessage(content="You are an expert competitive intelligence analyst. Extract data clearly based ONLY on the provided context."),
            HumanMessage(content=attr_content)
        ]
        
        try:
            attr_response = llm.invoke(attr_messages)
            gathered_data[attr] = attr_response.content
        except Exception as e:
            _log.error(f"    - Error extracting {attr} for {comp}: {e}")
            gathered_data[attr] = "Information extraction failed."
            
    return gathered_data


def synthesize_competitor_record(comp: str, gathered_data: Dict[str, str], llm) -> Optional[CompetitorInfo]:
    """Synthesize gathered raw data into a structured JSON record."""
    _log.info(f"  -> Synthesizing JSON record for {comp}...")
    synth_parser = PydanticOutputParser(pydantic_object=CompetitorInfo)
    
    synth_prompt = f"""
    You are a JSON formatting bot. Convert the following unstructured research about the game '{comp}' into the requested JSON schema.
    
    CRITICAL RULES:
    1. "features", "strengths", and "weaknesses" MUST be an array of strings. 
    2. If you cannot find any information for a list category, you MUST output an empty array []. Do NOT output strings like "None" or "No information".
    3. "price" MUST be a floating-point number (e.g. 0.0 for free, 9.99 for paid). If unknown, output null.
    4. "features" MUST NOT include any game genres (like RPG, MOBA, Battle Royale).
    
    RESEARCH:
    Features: {gathered_data.get('features', 'No information')}
    Pricing: {gathered_data.get('price', 'No information')}
    Strengths: {gathered_data.get('strengths', 'No information')}
    Weaknesses: {gathered_data.get('weaknesses', 'No information')}
    
    {synth_parser.get_format_instructions()}
    """
    
    try:
        synth_response = llm.invoke([HumanMessage(content=synth_prompt)])
        comp_record = synth_parser.parse(synth_response.content)
        _log.info(f"  ✅ Successfully compiled {comp}!")
        return comp_record
    except Exception as e:
        _log.error(f"  ❌ Failed to parse final JSON for {comp}: {e}")
        return None

    
def extract_competitor_intelligence(persist_directory: Path) -> List[CompetitorInfo]:
    """
    Extracts competitor intelligence from a vector database using a multi-stage pipeline.
    """
    _log.info("Connecting to DB and LLM...")
    try:
        store = get_vector_store(persist_directory=str(persist_directory))
        retriever = build_retriever(vector_store=store, initial_k=20, final_k=5)
        llm = get_llm()
    except Exception as e:
        _log.error(f"Failed to initialize store or LLM: {e}")
        return []
    
    # --- Discovery ---
    competitor_list = discover_competitors(store, llm)
    
    final_intelligence = []
    
    # --- Deep Analysis & Synthesis for each discovered competitor ---
    for comp in competitor_list:
        _log.info(f"\nEvaluating Competitor: {comp}...")
        
        # ---- Attribute Extraction ----
        gathered_data = analyze_competitor_attributes(comp, retriever, llm)
        
        # ---- Synthesis ----
        comp_record = synthesize_competitor_record(comp, gathered_data, llm)
        if comp_record:
            final_intelligence.append(comp_record)
            
    return final_intelligence


def main(persist_directory: Path, output_dir: Path):
    try:
        results = extract_competitor_intelligence(persist_directory=persist_directory)
        
        if not results:
            _log.warning("No intelligence records were extracted.")
            return

        _log.info("\n\n" + "="*50)
        _log.info("FINAL COMPETITIVE INTELLIGENCE REPORT")
        _log.info("="*50)
        
        output_data = [r.model_dump() for r in results]
        
        for r in output_data:
            print(json.dumps(r, indent=2))
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"competitor_intelligence.json"
        
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        _log.info(f"\nResults saved to {output_path}")

    except Exception as e:
        _log.error(f"An unexpected error occurred in main: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multi-Stage LLM Extraction Agent")
    parser.add_argument("--persist_directory", type=Path, default=Path("vector-db"), help="Directory containing the Chroma database.")
    parser.add_argument("--output_dir", type=Path, default=Path("json"), help="Directory to save the JSON results.")
    
    args = parser.parse_args()
    main(args.persist_directory, args.output_dir)
