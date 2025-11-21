# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from medical_bot import medical_chat

app = FastAPI()

chat_history = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(payload: dict):
    user_input = payload.get("message", "")
    reply = await medical_chat(user_input, chat_history)
    return { "reply": reply }
