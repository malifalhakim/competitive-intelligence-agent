import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = FastAPI(title="Competitive Intelligence Q&A API")

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

def setup_chain():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Missing Groq API Key (GROQ_API_KEY)")

    db = SQLDatabase.from_uri(f"sqlite:///database.db")

    llm = ChatGroq(model=MODEL_NAME, api_key=groq_api_key, temperature=0.1)

    write_query = create_sql_query_chain(llm, db)
    execute_query = QuerySQLDataBaseTool(db=db)

    answer_prompt = PromptTemplate.from_template(
    """Given the user question, the SQL query, and the SQL result, provide a clear and concise natural language answer.

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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
