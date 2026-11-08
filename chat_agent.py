import os
from pathlib import Path
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

@tool
def team_structure_tool(query: str) -> str:
    """This tool returns the team structure of the CTC team."""
    return {
        "teams":[
            {
                "name":"Management",
                "members":[
                    {
                        "name": "Anthony",
                        "role": "CIO"
                    },
                    {
                        "name": "Wang",
                        "role": "Site Lead"
                    },
                    {
                        "name": "Wang",
                        "role": "Technical Lead"
                    }
                ],
            },
            {
                "name": "Development",
                "members": [
                    {
                        "name": "Alice",
                        "role": "Developer"
                    },
                    {
                        "name": "Bob",
                        "role": "Developer"
                    },
                    {
                        "name": "Ken Chan",
                        "role": "Developer"
                    },
                    {
                        "name": "Ken Lai",
                        "role": "Baby"
                    },
                    {
                        "name": "Ken Lo",
                        "role": "Seafood"
                    }
                ]
            }
        ]
    }

@tool
def architecture_tool(query: str) -> str:
    """This tool returns the enterprise architecture details. Use this tool for queries related to the system architecture, software stack, and technology choices."""
    return {"architecture": "Microservices", "language": "Python"}

_chat_agent = None

def get_llm():
    print("No existing agent found, creating a new one.")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com/v1"
    )
    
    return llm

def _build_chat_agent():
    print("Building agent...")
    global _chat_agent
    if _chat_agent is not None:
        print("Agent already built, reusing existing agent.")
        return _chat_agent

    prompt_path = Path(__file__).parent / "system_prompt.md"
    system_prompt = prompt_path.read_text().strip()

    _chat_agent = create_agent(
        model=get_llm(),
        tools=[team_structure_tool, architecture_tool],
        system_prompt=system_prompt,
    )

    return _chat_agent