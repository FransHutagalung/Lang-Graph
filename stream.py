from typing import TypedDict , List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph , START , END
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatOllama(model="qwen:7b") 

def process(state: AgentState) -> AgentState:
    print("\nAI: ", end="", flush=True)
    full_response = ""
    
    for chunk in llm.stream(state["messages"]):
        content = chunk.content if chunk.content else ""
        print(content, end="", flush=True)
        full_response += content

    return {
        "messages": state["messages"] + [HumanMessage(content=full_response)]
    }

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("You: ")
agent.invoke({"messages": [HumanMessage(content=user_input)]})
