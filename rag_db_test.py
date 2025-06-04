from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from operator import add as add_messages
import sqlite3
import pandas as pd
import os
import json

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest", 
    temperature=0.1, 
    api_key=api_key
)

DATABASE_PATH = "sales_data.db"


def get_database_schema():
    """Get database schema information"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    schema_info = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        # PRAGMA table_info(nama_tabel) untuk mendapatkan informasi struktur kolom dalam tabel tersebut.
        columns = cursor.fetchall()
        schema_info[table_name] = [
            {
                'column': col[1], 
                'type': col[2], 
                'nullable': not col[3],
                'primary_key': bool(col[5])
            } 
            for col in columns
        ]
    
    conn.close()
    return schema_info

def execute_sql_query(query: str) -> List[Dict[str, Any]]:
    """Execute SQL query and return results"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        results = cursor.fetchall()
        
        # Convert to list of dictionaries
        result_list = []
        for row in results:
            result_dict = {}
            for i, value in enumerate(row):
                result_dict[columns[i]] = value
            result_list.append(result_dict)
        
        return result_list
    
    except Exception as e:
        return [{"error": str(e)}]
    
    finally:
        conn.close()

def generate_sql_from_natural_language(question: str, schema: Dict) -> str:
    """Generate SQL query from natural language using LLM"""
    schema_description = ""
    for table_name, columns in schema.items():
        schema_description += f"\nTable: {table_name}\n"
        for col in columns:
            schema_description += f"  - {col['column']} ({col['type']})"
            if col['primary_key']:
                schema_description += " [PRIMARY KEY]"
            schema_description += "\n"
    
    print(f"Schema Description: {schema_description}")
    prompt = f"""
    Based on the following database schema, convert the natural language question to a SQL query.
    
    Database Schema:
    {schema_description}
    
    Question: {question}
    
    Return ONLY the SQL query without any explanation or formatting.
    Make sure the query is syntactically correct for SQLite.
    Use proper JOINs when needed to access data from multiple tables.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    sql_query = response.content.strip()
    
    # Clean up the SQL query (remove markdown formatting if present)
    if sql_query.startswith('```sql'):
        sql_query = sql_query[6:]
    if sql_query.endswith('```'):
        sql_query = sql_query[:-3]
    
    return sql_query.strip()


schema = get_database_schema()

@tool
def database_query_tool(question: str) -> str:
    """
    Tool to query the sales database based on natural language questions.
    This tool can answer questions about customers, products, orders, and sales data.
    """
    try:
        # Generate SQL from natural language
        sql_query = generate_sql_from_natural_language(question, schema)
        print(f"Generated SQL: {sql_query}")
        
        # Execute the query
        results = execute_sql_query(sql_query)
        
        if not results:
            return "No data found for your query."
        
        if "error" in results[0]:
            return f"Database error: {results[0]['error']}"
        
        # Format results
        if len(results) == 1:
            return f"Query Result:\n{json.dumps(results[0], indent=2, default=str)}"
        else:
            formatted_results = "Query Results:\n"
            for i, result in enumerate(results[:10]):  # Limit to first 10 results
                formatted_results += f"\nResult {i+1}:\n{json.dumps(result, indent=2, default=str)}\n"
            
            if len(results) > 10:
                formatted_results += f"\n... and {len(results) - 10} more results"
            
            return formatted_results
    
    except Exception as e:
        return f"Error processing query: {str(e)}"

@tool
def database_schema_tool(table_name: str = "") -> str:
    """
    Tool to get database schema information.
    If table_name is provided, returns schema for that specific table.
    If no table_name is provided, returns schema for all tables.
    """
    try:
        if table_name and table_name in schema:
            table_info = schema[table_name]
            result = f"Schema for table '{table_name}':\n"
            for col in table_info:
                result += f"  - {col['column']} ({col['type']})"
                if col['primary_key']:
                    result += " [PRIMARY KEY]"
                result += "\n"
            return result
        else:
            result = "Database Schema:\n"
            for table_name, columns in schema.items():
                result += f"\nTable: {table_name}\n"
                for col in columns:
                    result += f"  - {col['column']} ({col['type']})"
                    if col['primary_key']:
                        result += " [PRIMARY KEY]"
                    result += "\n"
            return result
    
    except Exception as e:
        return f"Error getting schema: {str(e)}"

# Tools setup
tools = [database_query_tool, database_schema_tool]
llm = llm.bind_tools(tools)

# Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

# System prompt
system_prompt = """
You are an intelligent database assistant that helps users query and analyze sales data.
You have access to a sales database with the following tables:
- customers: customer information
- products: product catalog
- orders: order records
- order_items: detailed order items

You can use the following tools:
1. database_query_tool: Query the database using natural language questions
2. database_schema_tool: Get database schema information

Always try to provide helpful and accurate information based on the database content.
If you need to understand the database structure, use the schema tool first.
When presenting results, format them in a clear and readable way.
"""

tools_dict = {tool.name: tool for tool in tools}

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}

# Tool Execution Agent
def execute_tools(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""
    tool_calls = state['messages'][-1].tool_calls
    results = []
    
    for tool_call in tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        print(f"Calling Tool: {tool_name} with args: {tool_args}")
        
        if tool_name not in tools_dict:
            result = f"Error: Tool '{tool_name}' not found."
        else:
            try:
                result = tools_dict[tool_name].invoke(tool_args)
            except Exception as e:
                result = f"Error executing tool: {str(e)}"
        
        results.append(ToolMessage(
            tool_call_id=tool_call['id'],
            name=tool_name,
            content=str(result)
        ))
    
    print("Tools execution completed!")
    return {'messages': results}

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tools", execute_tools)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "tools", False: END}
)
graph.add_edge("tools", "llm")
graph.set_entry_point("llm")

# Compile the agent
database_rag_agent = graph.compile()

def run_database_agent():
    """Run the database RAG agent"""
    print("\n=== DATABASE RAG AGENT ===")
    print("Ask questions about customers, products, orders, and sales data!")
    print("Type 'exit' or 'quit' to stop.")
    
    # Show sample questions
    sample_questions = [
        "What is the database schema?",
        "Show me all customers from Jakarta",
        "What are the top selling products?",
        "Show me John Doe's order history",
        "What is the total sales amount?",
        "Which customers have pending orders?",
        "Show me products with low stock (less than 50)",
        "What is the average order value?"
    ]
    
    print("\nSample questions you can ask:")
    for i, question in enumerate(sample_questions, 1):
        print(f"{i}. {question}")
    
    while True:
        print("\n" + "="*50)
        user_input = input("\nYour question: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        try:
            messages = [HumanMessage(content=user_input)]
            result = database_rag_agent.invoke({"messages": messages})
            
            print("\n=== ANSWER ===")
            print(result['messages'][-1].content)
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    run_database_agent()