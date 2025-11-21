# medical_bot.py
from dotenv import load_dotenv
import os
import re

from langchain_community.vectorstores import FAISS

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

# -------------------------------
# Session memory will be passed from FastAPI, not stored here
# -------------------------------

# -------------------------------------------------------
# MEDICAL + GREETING + CASUAL DETECTION HELPERS
# -------------------------------------------------------
def is_medical_query(text: str) -> bool:
    casual_phrases = [
        "how are you", "what's up", "how is it going", "how do you do",
        "good morning", "good evening", "good afternoon", "good night",
        "thank you", "thanks", "please", "sorry", "excuse me",
        "what can you do", "who are you", "your name", "help me"
    ]
    
    text_lower = text.lower().strip()
    if any(phrase in text_lower for phrase in casual_phrases):
        return False
    
    medical_keywords = [
        "fever", "headache", "pain", "ache", "cough", "cold", "flu", "vomit", 
        "nausea", "rash", "bleeding", "fatigue", "weakness", "diarrhea", 
        "breathless", "chest pain", "stomach pain", "sore throat", "congestion",
        "dizziness", "swelling", "infection", "injury", "burn", "cut",
        "dengue", "malaria", "covid", "pneumonia", "asthma", "diabetes",
        "hypertension", "migraine", "arthritis", "allergy", "asthma",
        "medicine", "medication", "pill", "tablet", "dose", "prescription",
        "doctor", "hospital", "clinic", "emergency", "appointment",
        "symptom", "diagnosis", "treatment", "therapy", "recovery"
    ]
    
    words = text_lower.split()
    medical_word_count = sum(1 for word in words if word in medical_keywords)
    
    if medical_word_count >= 1 and len(words) > 2:
        return True
    
    symptom_patterns = [
        r"my\s+\w+\s+hurts", r"i\s+have\s+\w+\s+pain", r"i\s+feel\s+sick",
        r"i\s+have\s+fever", r"i\s+have\s+cough", r"my\s+\w+\s+aches",
        r"should\s+i\s+take", r"what\s+medicine", r"treatment\s+for"
    ]
    
    return any(re.search(pattern, text_lower) for pattern in symptom_patterns)


def is_greeting(text: str) -> bool:
    greetings = ["hi", "hello", "hey", "hola", "salam", "assalam", "hi there"]
    return text.lower().strip() in greetings


def is_casual_conversation(text: str) -> bool:
    casual_patterns = [
        "how are you", "what's up", "how is it going", "how do you do",
        "good morning", "good evening", "good afternoon", "good night",
        "thank you", "thanks", "please", "sorry", "excuse me",
        "what can you do", "who are you", "your name", "help me",
        "nice to meet you", "pleasure to meet you", "how old are you",
        "where are you from", "what time is it", "tell me about yourself"
    ]
    text_lower = text.lower().strip()
    return any(pattern in text_lower for pattern in casual_patterns)


# -------------------------------
# LOAD EMBEDDINGS & FAISS INDEX
# -------------------------------
use_openai = False
embedding_model = None
faiss_path = None
openai_client = None

if openai_key and len(openai_key) > 20:
    try:
        from openai import OpenAI
        from langchain_openai import OpenAIEmbeddings

        openai_client = OpenAI(api_key=openai_key)
        openai_client.embeddings.create(
            model="text-embedding-3-small",
            input="hello"
        )

        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        use_openai = True
        faiss_path = "medical_faiss_openai"
    except:
        pass

if not use_openai:
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_path = "medical_faiss_hf"

vector_store = FAISS.load_local(
    faiss_path,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

from groq import Groq
groq_client = Groq(api_key=groq_key)


def retrieve_relevant(query, k=20):
    results = vector_store.similarity_search(query, k=k)
    return " ".join([r.page_content for r in results])


# ============================================================
# MAIN FUNCTION CALLED BY FASTAPI
# ============================================================
async def medical_chat(user_input: str, chat_history: list):
    chat_history.append({"role": "user", "content": user_input})

    # 1. GREETING
    if is_greeting(user_input):
        return "Hello! 👋 I'm your medical assistant. How can I help you today?"

    # 2. CASUAL MODE
    if is_casual_conversation(user_input):
        prompt = f"""
You are a friendly assistant. Keep replies short and warm.

Conversation:
{chat_history}

User: {user_input}
"""
        try:
            if use_openai:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
        except:
            pass

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    # 3. MEDICAL MODE
    if is_medical_query(user_input):
        context = retrieve_relevant(user_input)

        prompt = f"""
You are a medical doctor. Use the context to give helpful, accurate advice.

Conversation:
{chat_history}

Medical Context:
{context}

Structure the answer as:
1. Symptoms Summary
2. Precautions
3. Safe Medicines / Treatments
4. When to Visit Hospital

Patient Input:
{user_input}
"""
        try:
            if use_openai:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
        except:
            pass

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    # 4. GENERAL MODE
    prompt = f"""
You are a helpful assistant:

Conversation: {chat_history}

User: {user_input}
"""

    try:
        if use_openai:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
    except:
        pass

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
