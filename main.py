from typing import TypedDict , List , Union
from langchain_core.messages import HumanMessage , AIMessage
from langgraph.graph import StateGraph , START , END
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama

import os

load_dotenv()

class AgentState(TypedDict) : 
    messages : List[Union[HumanMessage , AIMessage]]

llm = ChatOllama(model="qwen:7b")


def process(state : AgentState) -> AgentState :
    
    print(f"current state now {state['messages']}")
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    
    print(f"\nAI : {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process" , process)
graph.add_edge(START , "process")
graph.add_edge("process" , END)

agent = graph.compile()

conversation_history = []

user_input = input("You : ")
while user_input != "exit" :
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages" : conversation_history})
    # print(result["messages"][-1].content)
    conversation_history = result["messages"]
    user_input = input("You : ")
    
with open("logging.txt" , "w") as file : 
    file.write("Conversation History : \n")
    
    for message in conversation_history :  
        if(isinstance(message , HumanMessage)) :
            file.write(f"You : {message.content}\n")
        if(isinstance(message , AIMessage)) :
            file.write(f"AI : {message.content}\n")
    file.write("End of Conversation")
    
    