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

document_content = ""
# Define state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_calls: list
    
@tool
def update(content : str) -> str :
    """updates the document with provided content"""
    global document_content
    document_content = content
    return f"Document has been updated successfully ! the current cpntent is {document_content}"

@tool 
def save(filename : str) -> str : 
    """ Save the current document to a text file and finish the process 
    
    Args :
        filename : name for text file
    """
    
    global document_content
    if not filename.emdswith(".txt") : 
        filename = f"{filename}.txt"
    
    try : 
        with open(filename , "w") as file : 
            file.write(document_content)
            print(f"Document has been saved to {filename}")
            
    except Exception as e :
        return f"Error saving document : {str(e)}"
    
tools = [update , save]

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", api_key=api_key ).bind_tools(tools)


# def our_agent(state : AgentState) -> AgentState :
#     system_prompt = SystemMessage(content=f"""
#      You are a Drafter , a helpfull writiing assistant , you are going to help the user update and modify documents . 
     
#      - If the user wants to update and modify content , use the 'update' tool with the complete update content . 
#      - if the user wants finish , you nedd to use 'save' tool .
#      - Make sure to always show the current documment state after modifications .
     
#      The current document is : {document_content}                             
#     """
#     )
    
#     if not state["messages"] : 
#         user_input = "I am ready to help you update a document , what do you want to do ?"
#         user_message = HumanMessage(content=user_input)
#     else : 
#         user_input = input("\n What would you like to do ? ")
#         print(f"\n 🧙‍♂️  USER : {user_input}")
#         user_message = HumanMessage()
        
#     all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    
#     response = model.invoke(all_messages)
#     print(f"🥺 AI : {response.content}")
#     if hasattr(response , "tool_calls") and response.tool_calls :
#         print(f"🛠️ TOOL CALLS : {[tc['name'] for tc in response.tool_calls]}")
    
#     return {"messages" : list(state["messages"]) + [user_message , response]}

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are a Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to finish, you need to use the 'save' tool.
    - Always show the current document state after modifications.

    The current document is: {document_content}
    """)

    if not state["messages"]:
        user_input = "I am ready to help you update a document, what do you want to do?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do? ")
        print(f"\n🧙‍♂️ USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(all_messages)

    print(f"🥺 AI: {response.content}")
    tool_calls = getattr(response, "tool_calls", None)

    if tool_calls:
        print(f"🛠️ TOOL CALLS: {[tc['name'] for tc in tool_calls]}")
        return {
            "messages": list(state["messages"]) + [user_message, response],
            "tool_calls": tool_calls  # ini penting!
        }

    return {
        "messages": list(state["messages"]) + [user_message, response]
    }



def should_continue(state : AgentState) -> str : 
    """Determine if the conversation should continue or end."""
    
    messages = state["messages"]
    
    if not messages :  
        return "continue"
    
    for message in reversed(messages) : 
        if (isinstance(message , ToolMessage) and
            "saved" in message.content.lower() and
            "document" in message.content.lower()
        ): 
            return "end"
    
    return "continue"

def print_messages(messages) : 
    """Print messages."""
    
    for message in messages[-3:] : 
        if(isinstance(message , ToolMessage)) : 
            print(f"n 📍TOOL CALL  {message.content}" )
            
graph = StateGraph(AgentState)

graph.add_node("our_agent" , our_agent)
graph.add_node("tools" , ToolNode(tools=tools))

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "tools" , 
    should_continue ,
    {
        "continue" : "our_agent" , 
        "end" : END
    }
)

app = graph.compile()

def run_document_agent() : 
    print("\n =======DRAFTER=======")
    
    state = {"messages" : []}
    
    for step in app.stream(state , stream_mode="values") : 
        if "messages" in step :
            print_messages(step["messages"])
    
    print("\n ======DRAFTER FINISHED=======")
    
if __name__ == "__main__" : 
    run_document_agent()