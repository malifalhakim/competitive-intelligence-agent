import re
import os
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_log = logging.getLogger(__name__)

app = FastAPI(title="Competitive Intelligence Q&A API")

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
DB_PATH = "database.db"

def extract_sql(response: str) -> str:
    """Extracts and validates that only a SELECT SQL query is returned."""
    response = response.strip()
    
    sql_match = re.search(r"```(?:sql)?\s+(.*?)\s+```", response, re.DOTALL | re.IGNORECASE)
    clean_query = sql_match.group(1).strip() if sql_match else response
    
    clean_query = re.sub(r"^(SQLQuery|Question):", "", clean_query, flags=re.IGNORECASE).strip()

    upper_query = clean_query.upper()
    if not upper_query.startswith("SELECT"):
        _log.warning(f"Rejected non-SELECT query: {clean_query}")
        raise ValueError("Only SELECT queries are allowed for security reasons.")
    
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE", "ATTACH", "DETACH"]
    for word in forbidden:
        if f" {word} " in f" {upper_query} ":
             _log.warning(f"Rejected query containing forbidden keyword '{word}': {clean_query}")
             raise ValueError(f"Query contains forbidden keyword: {word}")

    return clean_query

def setup_chain():
    """
    Setup the LangChain chain for the Q&A system.
    """
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        _log.error("Missing Groq API Key (GROQ_API_KEY)")
        raise ValueError("Missing Groq API Key (GROQ_API_KEY)")

    db_uri = f"sqlite:///{DB_PATH}?mode=ro"
    db = SQLDatabase.from_uri(db_uri)

    llm = ChatGroq(model=MODEL_NAME, api_key=groq_api_key, temperature=0.1)

    write_query = create_sql_query_chain(llm, db) | RunnableLambda(extract_sql)
    execute_query = QuerySQLDataBaseTool(db=db)

    answer_prompt = PromptTemplate.from_template(
    """Given the user question, the SQL query, and the SQL result, provide a clear and concise natural language answer.
    If the result is empty, explain that no information was found for that specific question.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}

    Answer:"""
    )

    full_chain = (
        RunnablePassthrough.assign(query=write_query)
        | RunnablePassthrough.assign(result=lambda x: execute_query.invoke({"query": x["query"]}))
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    return write_query, execute_query, full_chain


write_query, execute_query, full_chain = setup_chain()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    sql_query: str
    sql_result: str
    answer: str

@app.post("/query", response_model=QueryResponse)
async def query_intelligence(request: QueryRequest):
    try:
        sql_query = write_query.invoke({"question": request.question})
        sql_result = execute_query.invoke({"query": sql_query})
        answer = full_chain.invoke({"question": request.question})

        return QueryResponse(
            question=request.question,
            sql_query=sql_query,
            sql_result=str(sql_result),
            answer=answer
        )

    except Exception as e:
        _log.error(f"Error processing query '{request.question}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
