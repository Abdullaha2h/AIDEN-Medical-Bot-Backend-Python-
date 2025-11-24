# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import asyncio
import logging

from medical_bot_pinecone import medical_chat

# ----------------- FastAPI setup -----------------
app = FastAPI(title="Medical Assistant Bot")

chat_history_store = {}  # store per user if needed (optional)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Request schema -----------------
class ChatPayload(BaseModel):
    message: str
    user_id: str = "default_user"  # optional for per-user chat history

# ----------------- Logging setup -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- Endpoint -----------------
@app.post("/chat")
async def chat(payload: ChatPayload):
    user_input = payload.message.strip()
    user_id = payload.user_id

    if user_id not in chat_history_store:
        chat_history_store[user_id] = []

    chat_history = chat_history_store[user_id]

    logger.info(f"User ({user_id}) input: {user_input}")

    try:
        # Await the async medical_chat function
        reply = await medical_chat(user_input, chat_history)
    except Exception as e:
        logger.error(f"Error in medical_chat: {e}")
        reply = "⚠️ Sorry, something went wrong while processing your request."

    chat_history.append({"role": "assistant", "content": reply})

    return {"reply": reply, "history": chat_history}

# ----------------- Test route -----------------
@app.get("/")
def root():
    return {"message": "Medical Assistant Bot is running. Use /chat endpoint to interact."}
