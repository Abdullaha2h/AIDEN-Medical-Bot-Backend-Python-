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

import time

class SessionStore:
    def __init__(self, max_history=20, ttl_seconds=3600):
        self.store = {}  # {user_id: {"history": [], "last_active": timestamp}}
        self.max_history = max_history
        self.ttl_seconds = ttl_seconds

    def get_history(self, user_id):
        self._cleanup()
        if user_id not in self.store:
            self.store[user_id] = {"history": [], "last_active": time.time()}
        
        # Update last active time
        self.store[user_id]["last_active"] = time.time()
        return self.store[user_id]["history"]

    def add_message(self, user_id, role, content):
        if user_id not in self.store:
            self.get_history(user_id)
        
        history = self.store[user_id]["history"]
        history.append({"role": role, "content": content})
        
        # Validation: Limit history size
        if len(history) > self.max_history:
             self.store[user_id]["history"] = history[-self.max_history:]
             
        self.store[user_id]["last_active"] = time.time()

    def _cleanup(self):
        """Remove sessions older than TTL"""
        now = time.time()
        expired_users = [
            uid for uid, data in self.store.items() 
            if now - data["last_active"] > self.ttl_seconds
        ]
        for uid in expired_users:
            del self.store[uid]

# Initialize the session store (1 hour timeout)
session_store = SessionStore(max_history=20, ttl_seconds=3600)

@app.post("/chat")
async def chat(payload: ChatPayload):
    user_input = payload.message.strip()
    user_id = payload.user_id

    # Get history and clean up old sessions
    chat_history = session_store.get_history(user_id)

    logger.info(f"User ({user_id}) input: {user_input}")

    try:
        # Await the async medical_chat function
        reply = await medical_chat(user_input, chat_history)
    except Exception as e:
        logger.error(f"Error in medical_chat: {e}")
        reply = "⚠️ Sorry, something went wrong while processing your request."

    # Update the store with the assistant's reply
    # This appends to the history list AND ensures limits/timestamps are updated
    session_store.add_message(user_id, "assistant", reply)

    return {"reply": reply, "history": chat_history}

# ----------------- Test route -----------------
@app.get("/")
def root():
    return {"message": "Medical Assistant Bot is running. Use /chat endpoint to interact."}
