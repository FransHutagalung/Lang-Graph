from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
import os

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

# Tool function
@tool
def add(a: int, b: int) -> int:
    """Adds two integers and returns the result."""
    return a + b

tools = [add]

# Initialize model and bind tools
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", api_key=api_key ).bind_tools(tools)

# Agent node
def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI Assistant, please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# Conditional check for whether to continue
def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    
    # Gemini tool invocation is usually in 'tool_calls' attribute
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

# Test input
inputs = {
    "messages": [HumanMessage(content="What is 2 + 3?")]
}

print_stream(agent.stream(inputs, stream_mode="values"))
