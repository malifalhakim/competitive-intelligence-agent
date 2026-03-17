# Competitor Intelligence Agent

A multi-stage LLM-powered system designed to extract deep competitive intelligence from game-related PDF documents. This project uses multi-modal extraction (text + tables + images) and provides a Natural Language to SQL interface for querying synthesized results.

## 🏗️ System Architecture

The system follows a modular pipeline to ensure high recall and structured data integrity:

1.  **Ingestion & Conversion**: Uses **Docling** to parse complex PDFs into structured JSON hierarchies, capturing text, tables (as HTML), and images (as Base64).
2.  **Chunking & Vectorization**: Chunks documents into semantic sections and stores them in **ChromaDB** using HuggingFace embeddings.
3.  **Collaborative Extraction (Agent)**:
    *   **Stage 1 (Discovery)**: Scans chunks iteratively to identify all game titles mentioned.
    *   **Stage 2-5 (deep Analysis)**: For each game, the agent performs specific retrieval-augmented generation (RAG) queries to extract Features, Pricing, Strengths, and Weaknesses.
4.  **Relational Synthesis**: Extracted JSON results are seeded into a **SQLite** database for structured analysis.
5.  **Interactive Q&A API**: A **FastAPI** backend that uses **LangChain's Text-to-SQL** capabilities to allow users to query the data in natural language.

## 🚀 How to Run with Docker

### 1. Prerequisites
Create a `.env` file in the root directory:
```env
HUGGINGFACEHUB_API_TOKEN=your_hf_token
GROQ_API_KEY=your_groq_key
JINA_API_KEY=your_jina_key (if using reranker)
```

### 2. Full Pipeline Execution
The process is designed as a sequential pipeline:

```bash
# Step 1: Convert PDFs in /dataset to structured JSON
docker compose run --rm docling-converter

# Step 2: Create semantic chunks for LangChain
docker compose run --rm langchain-chunker

# Step 3: Populate the Chroma database
docker compose run --rm vector-store

# Step 4: Run the Extraction Agent (This takes time/LLM calls)
docker compose run --rm agent

# Step 5: Seed the SQLite database
docker compose run --rm init_db

# Step 6: Start the Q&A API
docker compose up -d api
```

## 📥 Input & Output Example

### Input
A PDF market report (e.g., *State of Mobile Games in SEA 2024*) containing charts, game lists, and revenue tables.

### Output
The `agent` produces a `json/competitor_intelligence.json`:
```json
{
  "name": "Seven Knights Idle Adventure",
  "features": ["Character collection", "Auto-battle mechanics"],
  "price": 0.0,
  "strengths": ["Rapid revenue growth in first 3 months", "Strong IP leverage"],
  "weaknesses": ["Heavily reliant on the Korean market"]
}
```

### Interactive Query
**Request**: `POST /query` with `{"question": "Show me free games with strong IP?"}`
**Response**:
```json
{
  "answer": "Seven Knights Idle Adventure is a free-to-play game (0.0 cost) that leverages its strong Seven Knights IP to drive growth.",
  "sql_query": "SELECT name FROM Competitors JOIN Strengths ON ... WHERE price = 0.0 AND strength_text LIKE '%IP%'"
}
```

## ⚠️ Limitations

*   **Groq Rate Limits**: The iterative extraction agent makes many sequential calls. Ensure your Groq API tier allows for high RPM (Requests Per Minute).
*   **Context Window**: Extremely long documents with 100+ games may exceed the discovery stage's retrieval window (currently tuned to 50 chunks).
*   **SQL Generation**: While guarded by regex extraction, LLM-generated SQL can occasionally fail if the natural language question is ambiguous or uses column names not present in the schema.
*   **Data Integrity**: Features are extracted based *only* on the provided documents. If a document doesn't mention a price, the field will be `null`.

## 🛠️ Tech Stack
*   **LLM**: Llama 3/4 via Groq
*   **Orchestration**: LangChain (LCEL)
*   **Database**: ChromaDB (Vector) & SQLite (Relational)
*   **Parsing**: IBM Docling
*   **API**: FastAPI & Uvicorn
