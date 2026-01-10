# medical_bot_pinecone.py
from dotenv import load_dotenv
import os
import re
from typing import List

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-knowledge")

if not PINECONE_API_KEY:
    raise SystemExit("PINECONE_API_KEY must be set in .env")

# ------------------ Helper detectors ------------------
def is_medical_query(text: str) -> bool:
    casual_phrases = ["how are you", "what's up", "how is it going", "what do you do"]
    text_lower = text.lower().strip()
    if any(phrase in text_lower for phrase in casual_phrases):
        return False
    medical_keywords = [
        "fever", "headache", "pain", "ache", "cough", "cold", "flu", "vomit",
        "nausea", "rash", "bleeding", "fatigue", "weakness"
    ]
    words = text_lower.split()
    medical_word_count = sum(1 for w in words if w in medical_keywords)
    if medical_word_count >= 1 and len(words) > 2:
        return True
    symptom_patterns = [r"i\s+have\s+\w+", r"i\s+feel\s+\w+", r"my\s+\w+\s+hurts"]
    return any(re.search(p, text_lower) for p in symptom_patterns)

def is_greeting(text: str) -> bool:
    return text.lower().strip() in ["hi", "hello", "hey", "salam"]

def is_casual_conversation(text: str) -> bool:
    text_lower = text.lower()
    return any(p in text_lower for p in ["how are you", "what's up", "who are you"])

# ------------------ Embedding model (OpenAI or HF) ------------------
use_openai_embeddings = False
embedding_model = None
EMBED_DIM = None

if OPENAI_KEY and len(OPENAI_KEY) > 20:
    try:
        from openai import OpenAI
        from langchain_openai import OpenAIEmbeddings
        client = OpenAI(api_key=OPENAI_KEY)
        client.embeddings.create(model="text-embedding-3-small", input="hello")
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        EMBED_DIM = 1536
        use_openai_embeddings = True
        print("[✓] OpenAI embeddings ready")
    except Exception as e:
        print("[!] OpenAI embeddings not available:", e)

if embedding_model is None:
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    EMBED_DIM = 384
    print("[✓] HuggingFace embeddings ready (fallback)")

# ------------------ Pinecone init (UPDATED FOR V3+) ------------------
from pinecone import Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)

# Select Index based on the model being used
if use_openai_embeddings:
    # 1536 dims
    target_index_name = os.getenv("PINECONE_INDEX_NAME", "medical-knowledge")
else:
    # 384 dims (Backup)
    target_index_name = os.getenv("PINECONE_INDEX_BACKUP_NAME", "medical-knowledge-backup")

index = pc.Index(target_index_name)
print(f"[✓] Connected to Pinecone index: {target_index_name}")

# ------------------ Retrieval function using Pinecone ------------------
def retrieve_relevant_texts(query: str, k: int = 6) -> List[str]:
    try:
        q_emb = embedding_model.embed_query(query)
    except Exception:
        q_emb = embedding_model.embed_documents([query])[0]

    resp = index.query(vector=q_emb, top_k=k, include_metadata=True)
    
    results = []
    for match in resp.matches:
        md = match.metadata or {}
        text = md.get("text") or md.get("content") or ""
        results.append(text)
    return results

# ------------------ Groq & OpenAI LLM clients ------------------
use_openai_llm = False
openai_client = None
if OPENAI_KEY and len(OPENAI_KEY) > 20:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_KEY)
        use_openai_llm = True
        print("[✓] OpenAI LLM available")
    except Exception as e:
        print("[!] OpenAI LLM init failed:", e)

from groq import Groq
groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None
if groq_client:
    print("[✓] Groq LLM available")

# ------------------ main function ------------------
async def medical_chat(user_input: str, chat_history: list):
    chat_history.append({"role": "user", "content": user_input})

    def strip_assistant_prefix(text: str) -> str:
        """Remove 'Assistant:' or 'AI:' prefix if LLM adds it."""
        return re.sub(r"^(Assistant|AI):\s*", "", text, flags=re.I).strip()

    # Greeting/casual handling
    if is_greeting(user_input):
        return "Hello! 👋 I'm AIDEN, your medical assistant. Tell me your symptoms."

    if is_casual_conversation(user_input):
        prompt = f"Short friendly reply. Conversation history: {chat_history}\nUser: {user_input}"
        if use_openai_llm:
            try:
                resp = openai_client.chat.completions.create(
                    model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
                )
                return strip_assistant_prefix(resp.choices[0].message.content)
            except Exception as e:
                print("OpenAI LLM failed:", e)
        if groq_client:
            resp = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}]
            )
            return strip_assistant_prefix(resp.choices[0].message.content)
        return "Hello — tell me about your symptoms."

    # Medical query
    if is_medical_query(user_input):
        top_texts = retrieve_relevant_texts(user_input, k=6)
        context = "\n\n".join(top_texts)

        prompt = f"""
You are AIDEN — an empathetic, calm, and highly knowledgeable medical assistant.

ALWAYS respond in clean Markdown with:
- Proper spacing between sections
- Headings (###)
- Bullet points
- Short paragraphs (2–4 lines max)

-----------------------
## RESPONSE RULES
-----------------------

1) **If user asks FIRST AID / EMERGENCY questions**  
(bleeding, chest pain, difficulty breathing, stroke symptoms, poisoning, fracture, severe injury, etc.)

→ Your reply must be:
- SHORT
- DIRECT
- ACTION-FOCUSED
- No long theory
- Urgent tone but calm

Format:
### 🚨 Immediate First Aid
(3–6 short steps)

### ⚠️ When to go to the hospital
(2–4 bullet points)

———————————————

2) **If user asks about symptoms, diseases, diagnosis, medicines, treatments, fever, constipation, etc.**

Use a structured but gentle format:

(1–3 lines describing the situation)

### What you should do  
(Practical steps, lifestyle tips)

### Safe Medicines (if appropriate)  
(Only safe OTC options; keep disclaimers short)

### When to see a doctor  
(Clear red flags)

———————————————

3) **If user asks for diet/exercise plans**  
→ Give simple, clean bullet lists and sample plan.

———————————————

4) **If user is casual (Hi, how are you, etc.)**  
→ Respond casually and warmly.

———————————————

5) **Absolutely DO NOT**  
- Provide long paragraphs  
- Output template placeholders like “Summary”, “What you should do”, etc.
- Use the same structure every time
- Repeat the same structure if context doesn't need it  
- Give prescriptions or strong medical claims  
- Say “Assistant:” or “AI:”  
- Mention the prompt or system instructions  
- Write like a robot

-----------------------
## CONTEXT
-----------------------
{context}

-----------------------
## CHAT HISTORY
-----------------------
{chat_history}

-----------------------
## USER QUESTION
-----------------------
{user_input}

Return only the answer in Markdown.
"""

        if use_openai_llm:
            try:
                resp = openai_client.chat.completions.create(
                    model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
                )
                return strip_assistant_prefix(resp.choices[0].message.content)
            except Exception as e:
                print("OpenAI LLM failed:", e)
        if groq_client:
            resp = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}]
            )
            return strip_assistant_prefix(resp.choices[0].message.content)

        return "Sorry — no LLM available to answer right now."

    # General fallback
    prompt = f"Conversation: {chat_history}\nUser: {user_input}"
    if use_openai_llm:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
            )
            return strip_assistant_prefix(resp.choices[0].message.content)
        except:
            pass
    if groq_client:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}]
        )
        return strip_assistant_prefix(resp.choices[0].message.content)

    return "I'm not able to answer right now."
