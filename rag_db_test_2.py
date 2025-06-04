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
from datetime import datetime, date
import re

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
    """Get database schema information with sample data"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    schema_info = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # Get sample data to understand data format
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
        sample_rows = cursor.fetchall()
        
        schema_info[table_name] = {
            'columns': [
                {
                    'column': col[1], 
                    'type': col[2], 
                    'nullable': not col[3],
                    'primary_key': bool(col[5])
                } 
                for col in columns
            ],
            'sample_data': sample_rows
        }
    
    conn.close()
    return schema_info

def execute_sql_query(query: str) -> List[Dict[str, Any]]:
    """Execute SQL query and return results with better error handling"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Log the query for debugging
        print(f"Executing SQL: {query}")
        
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
        print(f"SQL Error: {str(e)}")
        return [{"error": str(e), "query": query}]
    
    finally:
        conn.close()

def get_current_date_info():
    """Get current date information for time-based queries"""
    now = datetime.now()
    return {
        'current_date': now.strftime('%Y-%m-%d'),
        'current_month': now.strftime('%Y-%m'),
        'current_year': str(now.year),
        'current_month_name': now.strftime('%B'),
        'current_month_number': str(now.month).zfill(2)
    }

def preprocess_indonesian_question(question: str) -> str:
    """Enhanced preprocessing for Indonesian questions with time context"""
    question_lower = question.lower()
    current_date_info = get_current_date_info()
    
    # Status mappings
    status_mappings = {
        'selesai': 'completed status',
        'sudah selesai': 'completed status',
        'complete': 'completed status',
        'done': 'completed status',
        'finish': 'completed status',
        'pending': 'pending status',
        'belum selesai': 'pending status',
        'menunggu': 'pending status',
        'dibatalkan': 'cancelled status',
        'batal': 'cancelled status',
        'cancelled': 'cancelled status',
        'dikirim': 'shipped status',
        'dalam pengiriman': 'shipped status',
        'shipped': 'shipped status'
    }
    
    # Time-based mappings
    time_mappings = {
        'bulan ini': f"this month ({current_date_info['current_month']})",
        'tahun ini': f"this year ({current_date_info['current_year']})",
        'hari ini': f"today ({current_date_info['current_date']})",
        'minggu ini': 'this week',
        'kemarin': 'yesterday'
    }
    
    # Replace Indonesian terms with English equivalents
    processed_question = question
    for indonesian_term, english_term in {**status_mappings, **time_mappings}.items():
        if indonesian_term in question_lower:
            processed_question = processed_question.replace(indonesian_term, english_term)
    
    # Add context hints
    if 'berapa' in question_lower:
        if 'pesanan' in question_lower or 'order' in question_lower:
            processed_question += " (count orders)"
        elif 'penjualan' in question_lower or 'sales' in question_lower:
            processed_question += " (sum total sales amount)"
        elif 'produk' in question_lower or 'product' in question_lower:
            processed_question += " (count products)"
    
    if 'siapa' in question_lower:
        processed_question += " (show customer names and details)"
    
    # Add date context for time-based queries
    if any(term in question_lower for term in ['bulan ini', 'tahun ini', 'hari ini']):
        processed_question += f" [Current date context: {current_date_info['current_date']}]"
    
    return processed_question

def generate_sql_from_natural_language(question: str, schema: Dict) -> str:
    """Enhanced SQL generation with better date handling"""
    current_date_info = get_current_date_info()
    
    schema_description = ""
    for table_name, table_info in schema.items():
        schema_description += f"\nTable: {table_name}\n"
        for col in table_info['columns']:
            schema_description += f"  - {col['column']} ({col['type']})"
            if col['primary_key']:
                schema_description += " [PRIMARY KEY]"
            schema_description += "\n"
        
        # Add sample data info
        if table_info['sample_data']:
            schema_description += f"  Sample data: {table_info['sample_data'][:2]}\n"
    
    # Enhanced status and date mapping
    context_info = f"""
    IMPORTANT CONTEXT INFORMATION:
    
    Current Date Information:
    - Today's date: {current_date_info['current_date']}
    - Current month: {current_date_info['current_month']} ({current_date_info['current_month_name']})
    - Current year: {current_date_info['current_year']}
    
    STATUS MAPPING for orders table:
    - The 'status' column contains values: 'Completed', 'Pending', 'Cancelled', 'Shipped'
    - When user asks about "selesai", "sudah selesai", "complete", "completed" → use WHERE status = 'Completed'
    - When user asks about "pending", "belum selesai", "menunggu" → use WHERE status = 'Pending'
    - When user asks about "dibatalkan", "cancelled", "batal" → use WHERE status = 'Cancelled'
    - When user asks about "dikirim", "shipped" → use WHERE status = 'Shipped'
    
    DATE FILTERING EXAMPLES:
    - For "bulan ini" (this month): WHERE strftime('%Y-%m', order_date) = '{current_date_info['current_month']}'
    - For "tahun ini" (this year): WHERE strftime('%Y', order_date) = '{current_date_info['current_year']}'
    - For "hari ini" (today): WHERE DATE(order_date) = '{current_date_info['current_date']}'
    
    COMMON QUERY PATTERNS:
    - Total sales: SELECT SUM(total_amount) FROM orders WHERE [conditions]
    - Count orders: SELECT COUNT(*) FROM orders WHERE [conditions]
    - Customer info: Use JOINs between customers and orders tables
    """
    
    prompt = f"""
    Based on the following database schema, convert the natural language question to a SQL query.
    
    Database Schema:
    {schema_description}
    
    {context_info}
    
    Question: {question}
    
    IMPORTANT INSTRUCTIONS:
    1. Return ONLY the SQL query without any explanation or formatting
    2. Make sure the query is syntactically correct for SQLite
    3. Use proper JOINs when needed to access data from multiple tables
    4. Pay special attention to date filtering using SQLite date functions
    5. For sales totals, use SUM(total_amount) from orders table
    6. Always use proper WHERE clauses for filtering
    7. Format numbers properly in results
    8. Use COALESCE or IFNULL for handling potential NULL values
    
    Examples:
    - "berapa total penjualan bulan ini" → SELECT COALESCE(SUM(total_amount), 0) as total_sales FROM orders WHERE strftime('%Y-%m', order_date) = '{current_date_info['current_month']}'
    - "berapa pesanan yang selesai" → SELECT COUNT(*) FROM orders WHERE status = 'Completed'
    - "siapa yang pesanannya pending" → SELECT DISTINCT c.name, c.email FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.status = 'Pending'
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    sql_query = response.content.strip()
    
    # Clean up the SQL query
    if sql_query.startswith('```sql'):
        sql_query = sql_query[6:]
    if sql_query.endswith('```'):
        sql_query = sql_query[:-3]
    
    return sql_query.strip()

# Initialize schema
schema = get_database_schema()

@tool
def database_query_tool(question: str) -> str:
    """
    Enhanced tool to query the sales database based on natural language questions.
    This tool can answer questions about customers, products, orders, and sales data.
    It understands Indonesian and English questions with time-based context.
    """
    try:
        # Preprocess question
        question_processed = preprocess_indonesian_question(question)
        
        # Generate SQL from natural language
        sql_query = generate_sql_from_natural_language(question_processed, schema)
        print(f"Original question: {question}")
        print(f"Processed question: {question_processed}")
        print(f"Generated SQL: {sql_query}")
        
        # Execute the query
        results = execute_sql_query(sql_query)
        
        if not results:
            return "Tidak ada data yang ditemukan untuk pertanyaan Anda."
        
        if "error" in results[0]:
            return f"Database error: {results[0]['error']}\nQuery: {results[0].get('query', 'N/A')}"
        
        # Enhanced result formatting
        if len(results) == 1:
            result = results[0]
            if len(result) == 1:
                # Single value result (like COUNT, SUM)
                key, value = next(iter(result.items()))
                if 'total' in key.lower() or 'sum' in key.lower():
                    return f"Total: {value:,}" if isinstance(value, (int, float)) else f"Total: {value}"
                elif 'count' in key.lower():
                    return f"Jumlah: {value}"
                else:
                    return f"{key}: {value}"
            else:
                return f"Hasil:\n{json.dumps(result, indent=2, default=str, ensure_ascii=False)}"
        else:
            formatted_results = f"Ditemukan {len(results)} hasil:\n"
            for i, result in enumerate(results[:10], 1):
                formatted_results += f"\n{i}. "
                if len(result) <= 3:
                    # Simple format for few columns
                    formatted_results += " | ".join([f"{k}: {v}" for k, v in result.items()])
                else:
                    formatted_results += json.dumps(result, indent=2, default=str, ensure_ascii=False)
                formatted_results += "\n"
            
            if len(results) > 10:
                formatted_results += f"\n... dan {len(results) - 10} hasil lainnya"
            
            return formatted_results
    
    except Exception as e:
        return f"Error memproses pertanyaan: {str(e)}"

@tool
def database_schema_tool(table_name: str = "") -> str:
    """Enhanced tool to get database schema information with sample data"""
    try:
        if table_name and table_name in schema:
            table_info = schema[table_name]
            result = f"Schema untuk tabel '{table_name}':\n"
            for col in table_info['columns']:
                result += f"  - {col['column']} ({col['type']})"
                if col['primary_key']:
                    result += " [PRIMARY KEY]"
                result += "\n"
            
            if table_info['sample_data']:
                result += f"\nContoh data:\n{table_info['sample_data'][:3]}"
            
            return result
        else:
            result = "Schema Database:\n"
            for table_name, table_info in schema.items():
                result += f"\nTabel: {table_name}\n"
                for col in table_info['columns']:
                    result += f"  - {col['column']} ({col['type']})"
                    if col['primary_key']:
                        result += " [PRIMARY KEY]"
                    result += "\n"
                
                if table_info['sample_data']:
                    result += f"  Contoh data: {table_info['sample_data'][0] if table_info['sample_data'] else 'Tidak ada data'}\n"
            
            return result
    
    except Exception as e:
        return f"Error mendapatkan schema: {str(e)}"

@tool
def get_current_date_tool() -> str:
    """Tool to get current date and time information"""
    date_info = get_current_date_info()
    return f"""Informasi Tanggal Saat Ini:
- Tanggal hari ini: {date_info['current_date']}
- Bulan ini: {date_info['current_month']} ({date_info['current_month_name']})
- Tahun ini: {date_info['current_year']}
"""

# Enhanced tools setup
tools = [database_query_tool, database_schema_tool, get_current_date_tool]
llm = llm.bind_tools(tools)

# Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

# Enhanced system prompt
system_prompt = f"""
Anda adalah asisten database yang cerdas untuk menganalisis data penjualan.
Anda memiliki akses ke database penjualan dengan tabel berikut:
- customers: informasi pelanggan (customer_id, name, email, city, country, registration_date)
- products: katalog produk (product_id, product_name, category, price, stock_quantity)
- orders: catatan pesanan (order_id, customer_id, order_date, total_amount, status)
- order_items: detail item pesanan (order_item_id, order_id, product_id, quantity, unit_price)

INFORMASI PENTING:
Tanggal hari ini: {get_current_date_info()['current_date']}
Bulan ini: {get_current_date_info()['current_month']}

Status pesanan dalam database menggunakan bahasa Inggris:
- 'Completed' = selesai, sudah selesai, complete, done
- 'Pending' = pending, belum selesai, menunggu
- 'Cancelled' = dibatalkan, cancelled, batal
- 'Shipped' = dikirim, shipped, dalam pengiriman

Tools yang tersedia:
1. database_query_tool: Query database menggunakan pertanyaan natural language
2. database_schema_tool: Mendapatkan informasi struktur database
3. get_current_date_tool: Mendapatkan informasi tanggal saat ini

Ketika pengguna bertanya tentang waktu (bulan ini, tahun ini, hari ini), gunakan informasi tanggal saat ini untuk memberikan hasil yang akurat.

Selalu berikan informasi yang membantu dan akurat berdasarkan konten database.
Jika perlu memahami struktur database, gunakan schema tool terlebih dahulu.
Format hasil dengan jelas dan mudah dibaca.
Respons dalam bahasa Indonesia jika pengguna bertanya dalam bahasa Indonesia.
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
    """Run the enhanced database RAG agent"""
    print("\n=== DATABASE RAG AGENT (ENHANCED) ===")
    print("Tanyakan tentang pelanggan, produk, pesanan, dan data penjualan!")
    print("Ketik 'exit' atau 'quit' untuk berhenti.")
    
    # Show current date info
    date_info = get_current_date_info()
    print(f"\nInformasi Tanggal: {date_info['current_date']} ({date_info['current_month_name']} {date_info['current_year']})")
    
    # Enhanced sample questions
    sample_questions = [
        "Apa schema database nya?",
        "Berapa total penjualan bulan ini?",
        "Berapa total penjualan tahun ini?",
        "Berapa banyak pesanan yang sudah selesai?",
        "Siapa saja pelanggan dari Jakarta?",
        "Produk apa yang paling laris?",
        "Siapa yang pesanannya masih pending?",
        "Berapa rata-rata nilai order?",
        "Produk apa yang stoknya kurang dari 50?",
        "Tampilkan riwayat pesanan John Doe",
        "Berapa total pesanan hari ini?",
        "Pelanggan mana yang paling banyak belanja?"
    ]
    
    print("\nContoh pertanyaan yang bisa ditanyakan:")
    for i, question in enumerate(sample_questions, 1):
        print(f"{i:2d}. {question}")
    
    while True:
        print("\n" + "="*60)
        user_input = input("\nPertanyaan Anda: ")
        
        if user_input.lower() in ['exit', 'quit', 'keluar']:
            print("Terima kasih! Sampai jumpa!")
            break
        
        try:
            messages = [HumanMessage(content=user_input)]
            result = database_rag_agent.invoke({"messages": messages})
            
            print("\n=== JAWABAN ===")
            print(result['messages'][-1].content)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Silakan coba pertanyaan lain.")

if __name__ == "__main__":
    run_database_agent()