from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from chat_agent import _build_chat_agent
from classification_agent import classification_chain, TextClassification

load_dotenv()
app = FastAPI()

@app.get("/")
def hello():
    return {"message": "Hello, Welcome to Simple Chatbot API"}


@app.post("/chat")
async def chat(payload: dict):
    input_text = payload.get("message", "")
    result = classification_chain.invoke({
        "text": input_text,
        "json_schema": TextClassification.model_json_schema()
    })
    print(result)
    
    agent = _build_chat_agent()

    # Extract the user's message and conversation history from the payload
    message = payload.get("message", "")
    history = payload.get("history", [])
    messages = []
    for turn in history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=message))

    # Invoke the agent with the constructed message list
    result = await agent.ainvoke({"messages": messages})
    msg_list = result.get("messages", [])

    # Get the last AI message as the response
    ai_response = ""
    for msg in reversed(msg_list):
        if isinstance(msg, AIMessage) and msg.content:
            ai_response = msg.content
            break

    # Build history from the full message list (human + AI turns)
    history = []
    for msg in msg_list:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            history.append({"role": "assistant", "content": msg.content})

    return {"response": ai_response, "history": history}
