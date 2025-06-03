from typing import Annotated, Sequence, TypedDict, List
from dotenv import load_dotenv
import time
import os
import requests
import json
import random
import datetime
from math import sqrt, pow, sin, cos, tan, pi

from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("GOOGLE_API_KEY not found in environment.")

# Define state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# =============== MATHEMATICAL TOOLS ===============
@tool
def calculator(expression: str) -> str:
    """
    Evaluasi ekspresi matematika yang aman.
    Contoh: "2 + 3 * 4", "sqrt(16)", "sin(3.14159/2)"
    """
    try:
        # Daftar fungsi yang diizinkan
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": sqrt,
            "sin": sin, "cos": cos, "tan": tan, "pi": pi,
            "__builtins__": {}
        }
        
        result = eval(expression, allowed_names)
        return f"Hasil dari {expression} = {result}"
    except Exception as e:
        return f"Error dalam perhitungan: {str(e)}"

@tool
def generate_fibonacci(n: int) -> List[int]:
    """Generate deret Fibonacci sampai n bilangan"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib

# =============== TEXT PROCESSING TOOLS ===============
@tool
def text_analyzer(text: str) -> dict:
    """Analisis teks: hitung kata, karakter, kalimat"""
    words = len(text.split())
    chars = len(text)
    chars_no_space = len(text.replace(' ', ''))
    sentences = len([s for s in text.split('.') if s.strip()])
    
    return {
        "jumlah_kata": words,
        "jumlah_karakter": chars,
        "karakter_tanpa_spasi": chars_no_space,
        "jumlah_kalimat": sentences,
        "rata_rata_kata_per_kalimat": round(words/sentences, 2) if sentences > 0 else 0
    }

@tool
def password_generator(length: int = 12, include_symbols: bool = True) -> str:
    """Generate password acak dengan panjang tertentu"""
    import string
    
    if length < 4:
        length = 4
    elif length > 50:
        length = 50
        
    chars = string.ascii_letters + string.digits
    if include_symbols:
        chars += "!@#$%^&*"
    
    password = ''.join(random.choice(chars) for _ in range(length))
    return f"Password yang dihasilkan: {password}"

# =============== DATE/TIME TOOLS ===============
@tool
def date_calculator(start_date: str, days_to_add: int) -> str:
    """
    Hitung tanggal setelah menambah/kurangi hari tertentu.
    Format tanggal: YYYY-MM-DD
    """
    try:
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        result_date = start + datetime.timedelta(days=days_to_add)
        
        return f"Tanggal {start_date} + {days_to_add} hari = {result_date.strftime('%Y-%m-%d (%A)')}"
    except ValueError:
        return "Format tanggal salah. Gunakan format: YYYY-MM-DD"

@tool
def get_current_time() -> str:
    """Mendapatkan waktu saat ini"""
    now = datetime.datetime.now()
    return f"Waktu saat ini: {now.strftime('%Y-%m-%d %H:%M:%S (%A)')}"

# =============== DATA PROCESSING TOOLS ===============
@tool
def list_statistics(numbers: List[float]) -> dict:
    """Hitung statistik dasar dari list angka"""
    if not numbers:
        return {"error": "List kosong"}
    
    sorted_nums = sorted(numbers)
    n = len(numbers)
    
    stats = {
        "jumlah_data": n,
        "minimum": min(numbers),
        "maksimum": max(numbers),
        "rata_rata": round(sum(numbers) / n, 2),
        "median": sorted_nums[n//2] if n % 2 == 1 else (sorted_nums[n//2-1] + sorted_nums[n//2]) / 2,
        "rentang": max(numbers) - min(numbers)
    }
    
    return stats

@tool
def word_frequency(text: str) -> dict:
    """Hitung frekuensi kata dalam teks"""
    words = text.lower().replace(',', '').replace('.', '').replace('!', '').replace('?', '').split()
    frequency = {}
    
    for word in words:
        if word in frequency:
            frequency[word] += 1
        else:
            frequency[word] = 1
    
    # Sort by frequency (descending)
    sorted_freq = dict(sorted(frequency.items(), key=lambda x: x[1], reverse=True))
    
    return {"frekuensi_kata": sorted_freq, "total_kata_unik": len(sorted_freq)}

# =============== UTILITY TOOLS ===============
@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Konversi satuan sederhana.
    Supported: celsius/fahrenheit, km/miles, kg/pounds, cm/inches
    """
    conversions = {
        ("celsius", "fahrenheit"): lambda x: (x * 9/5) + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        ("km", "miles"): lambda x: x * 0.621371,
        ("miles", "km"): lambda x: x / 0.621371,
        ("kg", "pounds"): lambda x: x * 2.20462,
        ("pounds", "kg"): lambda x: x / 2.20462,
        ("cm", "inches"): lambda x: x / 2.54,
        ("inches", "cm"): lambda x: x * 2.54,
    }
    
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {round(result, 4)} {to_unit}"
    else:
        return f"Konversi dari {from_unit} ke {to_unit} tidak didukung"

@tool
def random_quote_generator() -> str:
    """Generate quote motivasi acak"""
    quotes = [
        "Kesuksesan adalah perjalanan, bukan tujuan. - Ben Sweetland",
        "Cara terbaik untuk memulai adalah berhenti berbicara dan mulai melakukan. - Walt Disney",
        "Jangan takut gagal, takutlah tidak mencoba. - Unknown",
        "Hidup adalah 10% apa yang terjadi padamu dan 90% bagaimana kamu meresponnya. - Charles Swindoll",
        "Kesempatan tidak terjadi, kamu menciptakannya. - Chris Grosser",
        "Satu-satunya hal yang tidak mungkin adalah yang tidak pernah kamu coba. - Unknown",
        "Mimpi besar dan berani gagal. - Norman Vaughan"
    ]
    
    return random.choice(quotes)

# Daftar semua tools
tools = [
    calculator, generate_fibonacci, text_analyzer, password_generator,
    date_calculator, get_current_time, list_statistics, word_frequency,
    unit_converter, random_quote_generator
]

# Initialize model and bind tools
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest", 
    api_key=api_key
).bind_tools(tools)

# Agent node
def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="""
    Kamu adalah AI Assistant yang sangat membantu dengan berbagai tools praktis.
    
    Tools yang tersedia:
    1. calculator - untuk operasi matematika kompleks
    2. generate_fibonacci - membuat deret Fibonacci
    3. text_analyzer - analisis teks (kata, karakter, kalimat)
    4. password_generator - generate password aman
    5. date_calculator - kalkulasi tanggal
    6. get_current_time - waktu saat ini
    7. list_statistics - statistik dari list angka
    8. word_frequency - frekuensi kata dalam teks
    9. unit_converter - konversi satuan
    10. random_quote_generator - quote motivasi
    
    Gunakan tools ini untuk membantu user dengan berbagai kebutuhan mereka.
    Berikan penjelasan yang jelas dan helpful.
    """)
    
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# Conditional check for whether to continue
def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        return "continue"
    return "end"

# Graph construction
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
graph.add_edge("tools", "our_agent")

agent = graph.compile()

# Print streaming result
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if hasattr(message, "pretty_print"):
            message.pretty_print()
        else:
            print(message)

# =============== CONTOH PENGGUNAAN ===============
if __name__ == "__main__":
    print("=== LangGraph Multi-Tool Assistant ===\n")
    
    # Test cases
    test_cases = [
        "Hitung hasil dari sqrt(25) + sin(3.14159/2) * 10",
        "Generate 8 bilangan Fibonacci pertama",
        "Analisis teks ini: 'Python adalah bahasa pemrograman yang powerful dan mudah dipelajari'",
        "Buatkan password dengan panjang 15 karakter",
        "Berapa tanggal 30 hari setelah 2024-12-01?",
        "Hitung statistik dari angka: [12, 15, 18, 20, 22, 25, 28, 30]",
        "Konversi 100 fahrenheit ke celsius",
        "Berikan saya quote motivasi hari ini"
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test} ---")
        inputs = {"messages": [HumanMessage(content=test)]}
        print_stream(agent.stream(inputs, stream_mode="values"))
        time.sleep(5)
        print("-" * 50)