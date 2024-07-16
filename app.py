import json
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List      

import pandas as pd
from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel
from crewai_tools import tool
from langchain.schema.output import LLMResult
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_groq import ChatGroq
from langchain.memory import ConversationSummaryMemory
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


# Define the model for the request body
class PromptRequest(BaseModel):
    prompt: str
    session_id: str

# Define the model for creating a new session
class CreateSessionRequest(BaseModel):
    session_name: str

# Initialize FastAPI app
app = FastAPI()

df = pd.read_excel("spend_history_v1.2.xlsx")

# Convert columns to datetime format
# df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], format='%m/%d/%Y')
# df['contractstartdate'] = pd.to_datetime(df['contractstartdate'], format='%m/%d/%Y')
# df['contractenddate'] = pd.to_datetime(df['contractenddate'], format='%m/%d/%Y')
# df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'], format='%m/%d/%Y')

connection = sqlite3.connect("spend.db")
df.to_sql(name="PurchaseOrderCatalog", con=connection, if_exists='replace')

@dataclass
class Event:
    event: str
    timestamp: str
    text: str

def _current_time() -> str:
    return datetime.now(timezone.utc).isoformat()

class LLMCallbackHandler(BaseCallbackHandler):
    def __init__(self, log_path: Path):
        self.log_path = log_path

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        assert len(prompts) == 1
        event = Event(event="llm_start", timestamp=_current_time(), text=prompts[0])
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(event)) + "\n")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        generation = response.generations[-1][-1].message.content
        event = Event(event="llm_end", timestamp=_current_time(), text=generation)
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(event)) + "\n")

## Groq models

# llm = ChatGroq(
#     temperature=0,
#     model_name="llama3-70b-8192",  # mixtral-8x7b-32768, gemma2-9b-it, llama3-70b-8192
#     callbacks=[LLMCallbackHandler(Path("prompts.jsonl"))],
# )

# Gemini

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
#       temperature=0,
#       callbacks=[LLMCallbackHandler(Path("prompts.jsonl"))]
# )

# OpenAI

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    api_key= os.environ["OPENAI_API_KEY"],
    callbacks=[LLMCallbackHandler(Path("prompts.jsonl"))]
)

db = SQLDatabase.from_uri("sqlite:///spend.db")

@tool("list_tables")
def list_tables() -> str:
    """List the available tables in the database"""
    return ListSQLDatabaseTool(db=db).invoke("")

list_tables.run()

@tool("tables_schema")
def tables_schema(tables: str) -> str:
    """
    Input is a comma-separated list of tables, output is the schema and sample rows
    for those tables. Be sure that the tables actually exist by calling `list_tables` first!
    Example Input: table1, table2, table3
    """
    tool = InfoSQLDatabaseTool(db=db)
    return tool.invoke(tables)

print(tables_schema.run("PurchaseOrderCatalog"))

@tool("execute_sql")
def execute_sql(sql_query: str) -> str:
    """Execute a SQL query against the database. Returns the result"""
    return QuerySQLDataBaseTool(db=db).invoke(sql_query)

execute_sql.run("SELECT SUM(TotalCost) AS TotalSpend FROM PurchaseOrderCatalog WHERE SupplierName = 'Roberts-Ferguson' AND PurchaseDate >= DATE('now', '-2 months');")

@tool("check_sql")
def check_sql(sql_query: str) -> str:
    """
    Use this tool to double check if your query is correct before executing it. Always use this
    tool before executing a query with `execute_sql`.
    """
    return QuerySQLCheckerTool(db=db, llm=llm).invoke({"query": sql_query})

sql_dev = Agent(
    role="Senior SQLite Database Developer",
    goal="Construct and execute SQLite queries based on a request",
    backstory=dedent(
        """
        You are an experienced database developer who is master at creating efficient and complex SQLite queries.
        You have a deep understanding of how different databases work and how to optimize queries.
        You also have strong expertise in answering finanacial questions.
        Your skill is very good at creating SQLite queries which can fetch out the business questions.
        Be careful while working with TIMESTAMP type of column.
        **Before Using any tool first analyse Recent Chat History to answer the query. Do not call any tool if answer is already present in chat history**
        **Do not use CREATE TABLE query**
        You should work according to the following procedure:
        Step 1: Use the `list_tables` to find available tables.
        Step 2: Use the `tables_schema` to understand the metadata for the tables.
        **Always first check the columns names after using the tables_schema tool(be careful of the case of the column names)**
        Step 3: Use the `execute_sql` to check your queries for correctness.
        Step 4: Use the `check_sql` to execute queries against the database.
    """
    ),
    llm=llm,
    tools=[list_tables, tables_schema, execute_sql, check_sql],
    allow_delegation=False,
)

bus_analyst = Agent(
    role="Senior Business Analyst",
    goal="You receive data from the database developer and analyze it",
    backstory=dedent(
        """
                You have deep experience with analyzing datasets using Python.
                You have expertise in Multi-dimensional spend analysis.
                You have the following knowledge:
                Understanding of spend categories and hierarchies,
                Knowledge of various spend dimensions (e.g., supplier, department, product, geography), and
                Familiarity with financial metrics and KPIs.
            """

    ),
    llm=llm,
    allow_delegation=False,
)

report_writer = Agent(
    role="Senior Report Editor",
    goal="Write an executive summary type of report based on the work of the business analyst",
    backstory=dedent(
        """
       Your writing skill is well known for clear and effective communication.
        You always summarize long texts into bullet points that contain the most
        important details.
        In case the data received consists of more than four rows you always use tables or figures to present
        the data in a convenient manner.
        """
    ),
    llm=llm,
    allow_delegation=False,
)

extract_data = Task(
    description="""Extract data from the database that is required for the answering {query}.
    Create the queries in such a way that Business analyst can fetch out the information for the query.
    Never query for all the columns from a specific table, only ask for
    the relevant columns given the question.
    Always include the result of the SQL query in your task output""",
    expected_output="Database result for the query",
    agent=sql_dev,
)

analyze_data = Task(
    description="""Analyze the data from the database and write an analysis for {query}. Always include the Output of the Senior SQLite Database Developer agent in your task output. 
                        The final answer given by you should be to the point.
                        **Before giving any conclusion, first analyze Recent Chat History to answer the query.**""",
    expected_output='''Detailed analysis text in markdown format, In case the data received consists of more than four rows you always use tables or figures to present
                the data in a convenient manner.''',
    agent=bus_analyst,
    context=[extract_data]
        )


write_report = Task(
    description=dedent(
        """
        Write an comprehensive answer from the analysis for the given question : {query}.
        The answer should be within 100 words.
    """
    ),
    expected_output="Markdown report",
    agent=report_writer,
    context=[analyze_data],
)

crew = Crew(
    agents=[sql_dev, bus_analyst],
    tasks=[extract_data, analyze_data],
    process=Process.sequential,
    verbose=2,
    memory=False,
    output_log_file="crew.log",
)

memory = ConversationSummaryMemory(llm=llm)
session_history: Dict[str, Dict[str, Any]] = {}

@app.post("/query")
async def query_db(request: PromptRequest):
    session_id = request.session_id
    prompt = request.prompt
    context = (
        "Act as a Data Analyst'. "
        "There is the ONLY table in the database."
        "Given the above conversation generate a search query to lookup in order to get the information only relevant to the conversation."
        "Extract column names and table name and try to map user words with exact column names as user can use synonyms."
        "Use all the data and Run multiple queries if required before giving the final answer."
    )

    inputs = {
    "prompt": prompt
    }
    context_window = memory.load_memory_variables(inputs)
    conversation_context = f"Given the recent chat history {context_window['history']} , Answer the question: {inputs}."
    agent_input = {
        "query": conversation_context
    }

    try:
        response = crew.kickoff(inputs=agent_input)

        memory.save_context({"prompt": f"{inputs}"}, {"response": f"{response}"})

        # Save the conversation context externally
        if session_id not in session_history:
            session_history[session_id] = {"session_name": f"session{len(session_history) + 1}", "history": []}
        session_history[session_id]["history"].append({"role": "User", "message": prompt})
        session_history[session_id]["history"].append({"role": "EaseAI", "message": response})

        return {"response": response, "conversation": session_history[session_id]["history"]}
    except Exception as e:
        # Handling errors
        if "parsing error" in str(e).lower():
            clarifying_question = f"I encountered an error understanding your request: '{prompt}'. Can you please provide more details or clarify your question?"
            memory.save_context({"prompt": f"{prompt}"}, {"response": f"{clarifying_question}"})
            if session_id not in session_history:
                session_history[session_id] = {"session_name": f"session{len(session_history) + 1}", "history": []}
            session_history[session_id]["history"].append({"role": "User", "message": prompt})
            session_history[session_id]["history"].append({"role": "EaseAI", "message": clarifying_question})
            return {"response": clarifying_question, "conversation": session_history[session_id]["history"]}
        
        if "Error code: 429" in str(e).lower():
            clarifying_question = f"I encountered an Token Limit error Too much Content Passed. understanding your request: '{prompt}'. Can you please provide more details or clarify your question?"
            return {"response": clarifying_question}
        else:
            # Log the error for debugging
            print(f"Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset_memory")
async def reset_memory(session_id: str):
    memory.clear()
    if session_id in session_history:
        del session_history[session_id]
    return {"message": "Conversation memory reset successfully"}

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    return {"history": session_history.get(session_id, {}).get("history", [])}

@app.post("/create_session")
async def create_session(request: CreateSessionRequest):
    session_id = str(uuid.uuid4())
    session_name = f"Mohan {len(session_history) + 1}"
    session_history[session_id] = {"session_name": session_name, "history": []}
    return {"session_id": session_id, "session_name": session_name}

@app.get("/sessions")
async def get_sessions():
    return {"sessions": [{"session_id": sid, "session_name": sdata["session_name"]} for sid, sdata in session_history.items()]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
