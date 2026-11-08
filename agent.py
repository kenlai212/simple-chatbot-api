import os
from pathlib import Path
from click import prompt
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal

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

_agent = None

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

def _build_agent():
    print("Building agent...")
    global _agent
    if _agent is not None:
        print("Agent already built, reusing existing agent.")
        return _agent

    prompt_path = Path(__file__).parent / "system_prompt.md"
    system_prompt = prompt_path.read_text().strip()

    _agent = create_agent(
        model=get_llm(),
        tools=[team_structure_tool, architecture_tool],
        system_prompt=system_prompt,
    )

    return _agent


class UserIntent(BaseModel):
    """Categorize the user's input to determine the next workflow step."""
    
    intent: Literal["team_structure", "enterprise_architecture", "insurance_inquiry", "general_chitchat"] = Field(
        description="The primary intent of the user's message."
    )
    confidence: float = Field(
        description="Confidence score for this classification between 0.0 and 1.0."
    )
    reasoning: str = Field(
        description="A brief explanation of why this intent was chosen."
    )

"""
def classify_user_intent(user_input: str) -> UserIntent:
    # Create a clear classification prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an advanced intent classification routing bot. Analyze the user's input and classify it accurately."),
        ("human", "{user_input}")
    ])

    # Combine components into a structured classifier chain
    classifier = prompt | get_llm().with_structured_output(UserIntent)

    #Classify the user's intent based on the input string.
    return classifier.invoke({"user_input": user_input})
"""