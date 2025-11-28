# 🩺 AIDEN Medical AI Assistant – FastAPI + RAG (OpeanAI + Groq (Backup) + Hugging Face) + Next.js

🔗 **Live Preview:**  
https://aidenpro.vercel.app/

A lightweight **AI-powered medical assistant** built using:

- **FastAPI** (Python backend)
- **RAG pipeline** (Retrieval-Augmented Generation)
- **OpenAI / Groq LLM as backup**
- **Hugging face and faiss for embeddings**
- **Next.js frontend** with chat UI
- **Markdown-rendered responses**
- **Pinecone** to save memory on deployment

This project is designed for portfolios, demos, and learning full-stack AI integration.

---

## 🚀 Tech Stack (Frontend)

- **React**
- **TypeScript**
- **Tailwind CSS**
- **Shadcn/UI** components
- **Lucide React Icons**
- **Dark / Light mode** with Theme Provider
- **Responsive UI**

---

## 🚀 Features

- AI-generated medical guidance  
- Works with **OpenAI**, **Groq**, or fallback LLMs  
- RAG support (PDFs / embeddings / vector search)  
- REST API `/chat` using FastAPI  
- Fully responsive chat interface (Next.js + Tailwind)  
- Typing indicator + markdown rendering  
- Clean project structure

---


---

## 🔧 Installation

### **1️⃣ Create & activate virtual environment**

```
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### **2️⃣ Install dependencies**

```
pip install -r requirements.txt
```

### **3️⃣ Run FastAPI backend**

```
uvicorn server:app --host 0.0.0.0 --port 8000
```

Backend will now run at:

```
http://localhost:8000
```

---

## 🌐 API Usage

### **POST /chat**

Send a message + history:

```
POST http://localhost:8000/chat
Content-Type: application/json
```

Body:

```json
{
  "message": "I have fever and body pain",
  "history": []
}
```

Response:

```json
{
  "reply": "Your symptoms suggest..."
}
```

---

## 🌍 Deployment (Render)

### **1️⃣ Add new Web Service**
- Runtime: **Python 3.11+**
- Start Command:

```
uvicorn server:app --host 0.0.0.0 --port $PORT
```

### **2️⃣ Add environment variables**
```
OPENAI_API_KEY=xxxx
GROQ_API_KEY=xxxx
```

---

## 🤝 Next.js Frontend Integration

Your frontend sends messages using:

```
POST /api/chat → calls FastAPI backend
```

Example:

```ts
const res = await fetch("https://your-render-url/chat", {
  method: "POST",
  body: JSON.stringify({ message, history }),
});
```

---

Feel free to contribute or open issues!

