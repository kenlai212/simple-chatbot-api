import os
from pathlib import Path
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate

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

@tool
def insurance_products_tool(query: str) -> str:
    """This tool returns the insurance products offered by the company."""
    return {
        "products": [
            {
                "name": "Life Insurance",
                "description": "Provides financial protection for your loved ones."
            },
            {
                "name": "Health Insurance",
                "description": "Covers medical expenses and health-related costs."
            },
            {
                "name": "Retirement Plans",
                "description": "Helps you save for a secure retirement."
            }
        ]
    }

@tool
def design_theme_tool(query: str) -> str:
    """This tool returns the design themes used in the company."""
    return {
        "themes": [
            {
                "name": "Modern",
                "description": "A sleek and contemporary design style."
            },
            {
                "name": "Classic",
                "description": "A timeless and elegant design style."
            },
            {
                "name": "Minimalist",
                "description": "A clean and simple design style."
            }
        ]
    }

_chat_agent = None
_classification_chain = None

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

def _build_chat_agent(inputText):
    print("Building agent...")

    # Perform text classification on the input text to determine its category
    classificationResult = _build_classification_chain().invoke({
        "text": inputText,
        "json_schema": TextClassification.model_json_schema()
    })
    category = classificationResult.category
    print("Classification result:", classificationResult.category)

    #map classification result to tools
    if category == "team_structure":
        tools = [team_structure_tool]
    elif category == "enterprise_architecture":
        tools = [architecture_tool]
    elif category == "insurance_products":
        tools = [insurance_products_tool]
    elif category == "design_theme":
        tools = [design_theme_tool]
    else:
        tools = []

    global _chat_agent
    if _chat_agent is not None:
        print("Agent already built, reusing existing agent.")
        return _chat_agent

    # Read the chat system prompt from the markdown file
    chat_prompt_path = Path(__file__).parent / "chat_system_prompt.md"
    chat_system_prompt = chat_prompt_path.read_text().strip()

    _chat_agent = create_agent(
        model=get_llm(),
        tools=tools,
        system_prompt=chat_system_prompt,
    )

    return _chat_agent

def _build_classification_chain():
    global _classification_chain
    if _classification_chain is not None:
        return _classification_chain

    # Bind the Pydantic schema to the model using structured output
    structured_llm = get_llm().with_structured_output(TextClassification, method="json_mode")

    # Read the chat system prompt from the markdown file
    classification_prompt_path = Path(__file__).parent / "classification_system_prompt.md"
    classification_system_prompt = classification_prompt_path.read_text().strip()
    
    # Create a prompt template
    classification_prompt = ChatPromptTemplate.from_messages([
        (
            "system", classification_system_prompt
        ),
        ("user", "{text}")
    ])

    # Combine the prompt template and structured LLM into a classification chain
    _classification_chain = classification_prompt | structured_llm
    return _classification_chain


# Define the desired structured output schema using Pydantic
class TextClassification(BaseModel):
    category: Literal["team_structure", "enterprise_architecture", "insurance_products", "design_theme", "general_inquiry"] = Field(
        description="The primary category of the user's input."
    )
    confidence: float = Field(
        description="Confidence score of the classification between 0.0 and 1.0."
    )
    reasoning: str = Field(
        description="Brief explanation of why this category was chosen."
    )