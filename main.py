from typing import TypedDict , List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph , START , END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

import os

load_dotenv()

class AgentState(TypedDict) : 
    messages : List[HumanMessage]

llm = ChatOpenAI(
    model="qwen/qwen3-30b-a3b",  # ID model dari OpenRouter
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def process(state : AgentState) -> AgentState :
    response = llm.invoke(state["messages"])
    print(f"\nAI : {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process" , process)
graph.add_edge(START , "process")
graph.add_edge("process" , END)

agent = graph.compile()

user_input = input("You : ")
agent.invoke({"messages": [HumanMessage(content=user_input)]})
print(agent)