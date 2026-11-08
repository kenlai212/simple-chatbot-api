import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal

# Load environment variables from .env
load_dotenv()

# Define the desired structured output schema using Pydantic
class TextClassification(BaseModel):
    category: Literal["team_structure", "enterprise_architecture", "insurance_products", "general_inquiry"] = Field(
        description="The primary category of the user's input."
    )
    confidence: float = Field(
        description="Confidence score of the classification between 0.0 and 1.0."
    )
    reasoning: str = Field(
        description="Brief explanation of why this category was chosen."
    )

# Initialize the DeepSeek model (deepseek-chat supports structured output)
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.0,  # Low temperature for deterministic classification
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# Bind the Pydantic schema to the model using structured output
structured_llm = llm.with_structured_output(TextClassification, method="json_mode")

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert customer service triage agent. Analyze the incoming text and classify it accurately. You must always respond with a valid JSON object matching this schema: {json_schema}"
    ),
    ("user", "{text}")
])

# Combine prompt and structured LLM into a chain
classification_chain = prompt | structured_llm