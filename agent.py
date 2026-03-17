import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

from vector_store import get_vector_store, build_retriever


class CompetitorInfo(BaseModel):
    name: str = Field(description="The name of the competitor company or product.")
    features: List[str] = Field(description="A list of core features or capabilities offered by this competitor.")
    price: Optional[str] = Field(description="Details regarding the pricing, cost, or monetization model of the competitor.")
    strengths: List[str] = Field(description="Key strengths, advantages, or unique selling propositions (USPs).")
    weaknesses: List[str] = Field(description="Key weaknesses, limitations, or vulnerabilities (SWOT).")

class InitialDiscoveryList(BaseModel):
    competitors: List[str] = Field(description="A comprehensive list of all competitor names found in the context.")


def get_llm():
    """Initializes the Qwen3-32B model via Groq API."""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Missing Groq API Key (GROQ_API_KEY)")
        
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=groq_api_key,
        max_tokens=8192,
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

    
def extract_competitor_intelligence(persist_directory: Path):
    """
    Extracts competitor intelligence from a vector database using a multi-stage pipeline.
    """
    print("Connecting to DB and LLM...")
    store = get_vector_store(persist_directory=str(persist_directory))
    retriever = build_retriever(vector_store=store, initial_k=20, final_k=5)
    llm = get_llm()
    

    print("\nExecuting Stage 1: Competitor Discovery...")
    discovery_query = "List all product names mentioned in the documents"
    discovery_docs = retriever.invoke(discovery_query)
    
    discovery_parser = PydanticOutputParser(pydantic_object=InitialDiscoveryList)
    
    discovery_content = format_context_for_multimodal(discovery_docs)
    discovery_content.append({"type": "text", "text": f"\n\nBased ONLY on the context above, answer the query: '{discovery_query}'.\n\n{discovery_parser.get_format_instructions()}"})
    
    messages = [
        SystemMessage(content="You are an expert competitive intelligence analyst. Extract information accurately and follow the JSON format instructions perfectly."),
        HumanMessage(content=discovery_content)
    ]

    print(messages)
    
    discovery_response = llm.invoke(messages)
    try:
        clean_response = discovery_response.content
        competitor_list = discovery_parser.parse(clean_response).competitors
        print(f"Competitors Discovered: {competitor_list}")
    except Exception as e:
        print(f"Parsing failed for Discovery Stage. Raw output:\n{discovery_response.content}")
        return []


    final_intelligence = []
    
    for comp in competitor_list:
        print(f"\nEvaluating Competitor: {comp}...")
        
        gathered_data = {}
        
        queries = {
            "features": f"What are the core features, genres, or capabilities of the competitor '{comp}'?",
            "price": f"What is the pricing, cost, or monetization model of the competitor '{comp}'?",
            "strengths": f"What are the key strengths and advantages of the competitor '{comp}'?",
            "weaknesses": f"What are the weaknesses or limitations of the competitor '{comp}'?"
        }
        
        for attr, query in queries.items():
            print(f"  -> Retrieving & Analyzing: {attr.upper()}")
            attr_docs = retriever.invoke(query)
            
            attr_content = format_context_for_multimodal(attr_docs)
            attr_content.append({"type": "text", "text": f"\n\nBased ONLY on the visual and text context above, comprehensively answer the following query: '{query}'. If the information is not present, state 'No information available.'"})
            
            attr_messages = [
                SystemMessage(content="You are an expert competitive intelligence analyst. Extract data clearly based ONLY on the provided context."),
                HumanMessage(content=attr_content)
            ]
            
            attr_response = llm.invoke(attr_messages)
            gathered_data[attr] = attr_response.content
            
        print(f"  -> Synthesizing JSON record for {comp}...")
        synth_parser = PydanticOutputParser(pydantic_object=CompetitorInfo)
        
        synth_prompt = f"""
        You are a JSON formatting bot. Convert the following unstructured research about '{comp}' into the requested JSON schema.
        
        RESEARCH:
        Features: {gathered_data['features']}
        Pricing: {gathered_data['price']}
        Strengths: {gathered_data['strengths']}
        Weaknesses: {gathered_data['weaknesses']}
        
        {synth_parser.get_format_instructions()}
        """
        
        synth_response = llm.invoke([HumanMessage(content=synth_prompt)])
        
        try:
            comp_record = synth_parser.parse(synth_response.content)
            final_intelligence.append(comp_record)
            print(f"  ✅ Successfully compiled {comp}!")
        except Exception as e:
            print(f"  ❌ Failed to parse final JSON for {comp}.")
            print(synth_response.content)
            
    return final_intelligence


if __name__ == "__main__":
    import datetime
    parser = argparse.ArgumentParser(description="Run Multi-Stage LLM Extraction Agent")
    parser.add_argument("--persist_directory", type=Path, default=Path("vector-db"), help="Directory containing the Chroma database.")
    parser.add_argument("--output_dir", type=Path, default=Path("json"), help="Directory to save the JSON results.")
    
    args = parser.parse_args()
    
    results = extract_competitor_intelligence(persist_directory=args.persist_directory)
    
    print("\n\n" + "="*50)
    print("FINAL COMPETITIVE INTELLIGENCE REPORT")
    print("="*50)
    
    output_data = [r.model_dump() for r in results]
    
    for r in output_data:
        print(json.dumps(r, indent=2))
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"competitor_intelligence_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")
