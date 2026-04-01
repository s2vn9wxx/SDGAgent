from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    
    store_code: str
    analysis_result: str
    marketing_strategy: str
    
    next_step: str
    next_step_details: str
    final_answer: str