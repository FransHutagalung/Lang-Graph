from typing import Annotated , Sequence , TypedDict 
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage 
from langchain_core.messages import ToolMessage 
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.chat_models import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph , START , END
from langgraph.prebuilt import ToolNode
import os


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

class AgentState(TypedDict) :
    messages : Annotated[Sequence[BaseMessage] , add_messages]
    
@tool
def add(a : int , b : int) -> int :
    """Adds two integers and returns the result."""
    return a + b

tools = [add]

model = ChatGoogleGenerativeAI(model="gemini-pro", api_key=api_key).bind_tools(tools)

def model_call(state : AgentState) -> AgentState :
    system_prompt = SystemMessage(content="You are my AI Assistant , please answer my query to the best your ability")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages" : [response]} 

def should_continue(state : AgentState) -> AgentState :
   message = state["messages"]
   last_message = message[-1]
   if not last_message.tool_calls :
     return "end"
   else : 
     return "continue"
   

graph = StateGraph(AgentState)
graph.add_node('our_agent' , model_call)

tool_node = ToolNode(tools=tools)
graph.add_node('tools' , tool_node)

graph.set_entry_point('our_agent')
graph.add_conditional_edges(
    'our_agent' , 
    should_continue , 
    {
        'continue' : 'tools' , 
        'end' : END
    }
)


graph.add_edge('tools' , 'our_agent')

agent = graph.compile()

def print_stream(stream) : 
    for s in stream :
        message = s["message"][-1]
        if isinstance(message , tuple) :
            print(message)
        else :
            message.pretty_print()

inputs = {
    "messages": [HumanMessage(content="What is 2 + 3?")]
}
# print_stream(agent.stream(inputs , stream_mode = "values"))
print_stream(agent.stream(inputs, stream_mode="values"))