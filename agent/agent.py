import os
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from .tools import ingest, rag_tool, wolfram_tool

VDB = ingest()

def rag_math_tool(query: str):
    return rag_tool(query, VDB)

def wolfram_call(expr: str):
    return wolfram_tool(expr)

agent = LlmAgent(
    model="gemini-2.5-flash",
    name="calculus_agent",
    tools=[rag_math_tool, wolfram_call],
)
runner = Runner(agent=agent, session_service=InMemorySessionService())

def chat():
    user = input("You: ")
    resp = runner.run(user_id="me", message=user)
    print(resp)
