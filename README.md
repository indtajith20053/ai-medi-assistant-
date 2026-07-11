# 🩺 Medical AI Assistant

A multi-agent, Retrieval-Augmented Generation (RAG) medical information assistant built with **LangGraph**, **ChromaDB**, and **Gradio**. It routes user questions to specialist agents, grounds its medical knowledge in a WHO reference document, and includes deterministic safety guardrails for emergency situations.

> ⚠️ **Disclaimer:** This assistant provides general educational information only. It does not diagnose conditions or prescribe treatment. Always consult a qualified healthcare provider.

---

## ✨ Features

- **Multi-agent architecture** — a Supervisor agent routes each message to one of four specialists:
  - 🩹 **Symptom Assessment Agent** — educational, non-diagnostic symptom triage
  - 📚 **Medical Knowledge (RAG) Agent** — answers grounded in a WHO reference PDF
  - 💊 **Medication Information Agent** — general drug info, no dosage advice
  - 💬 **General Agent** — greetings and small talk (no LLM call, saves free-tier quota)
- **Retrieval-Augmented Generation** using ChromaDB + `sentence-transformers/all-MiniLM-L6-v2` embeddings with **Maximal Marginal Relevance (MMR)** retrieval for diverse, relevant context
- **Deterministic emergency detection** — keyword-based safety net that runs *before* any LLM call, so it can never be bypassed by a prompt
- **Automatic model fallback** — if a free OpenRouter model is rate-limited or down, the app automatically retries with a fallback model
- **Non-negotiable safety rules** baked into every agent's system prompt (no diagnoses, no dosage instructions, always recommend professional care)
- Built entirely on **free-tier tools**: OpenRouter free models, HuggingFace embeddings, ChromaDB (local/persisted), Gradio UI

---

## 🏗️ Architecture

```
User Message
     │
     ▼
Emergency Keyword Check (deterministic) ──► Emergency response (if triggered)
     │
     ▼
Supervisor Agent (routes to one specialist)
     │
     ├─► Symptom Assessment Agent
     ├─► Medical Knowledge Agent (RAG: ChromaDB + MMR retrieval)
     ├─► Medication Information Agent
     └─► General Agent (no LLM call)
     │
     ▼
Response (with safety disclaimers)
```

Built with **LangGraph** as a state machine (`StateGraph`), where each agent is a graph node and the Supervisor's routing decision determines the conditional edge taken.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph |
| LLMs | OpenRouter (free-tier models: Llama 3.3 70B, GPT-OSS 120B, Nemotron Ultra) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (persisted locally) |
| PDF Processing | LangChain `PyPDFLoader` + `RecursiveCharacterTextSplitter` |
| UI | Gradio (`ChatInterface`) |
| Framework | LangChain (`langchain-openai`, `langchain-community`, `langchain-huggingface`, `langchain-chroma`) |

---

## 📁 Project Structure

```
.
├── app.py                     # Main application (entry point)
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container build (for Fly.io / Railway / any Docker host)
├── fly.toml                   # Fly.io deployment config
├── .dockerignore
├── knowledge_base/
│   └── who_guideline.pdf      # WHO reference document (RAG knowledge source)
└── chroma_db/                 # Auto-generated vector store (persisted after first run)
```

---

## 🚀 Setup & Local Run

### 1. Clone the repo
```bash
git clone https://github.com/indtajith20053/ai-medi-assistant-.git
cd ai-medi-assistant-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API key
Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```
Get a free key at [openrouter.ai](https://openrouter.ai/settings/keys).

### 4. Add the knowledge base
Place a WHO (or other authoritative medical) reference PDF at:
```
knowledge_base/who_guideline.pdf
```
On first run, the app will automatically load, chunk, embed, and persist it to ChromaDB. Subsequent runs reuse the persisted vector store.

### 5. Run the app
```bash
python app.py
```
The Gradio interface will launch locally (default: `http://localhost:7860`).

---

## ☁️ Deployment

This app is container-ready and has been deployed on:


- **Railway** (GitHub auto-deploy)
  

For any platform, set the following environment variable/secret:

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | Your OpenRouter API key (required) |

The app respects `GRADIO_SERVER_NAME` and `GRADIO_SERVER_PORT` environment variables (defaults: `0.0.0.0` and `7860`), making it portable across hosting platforms without code changes.

---

## 🔒 Safety Design

- **Never diagnoses** — agents are explicitly instructed to avoid naming specific conditions
- **Never recommends dosages** — medication agent redirects all dosage questions to a doctor/pharmacist
- **Emergency short-circuit** — a deterministic keyword check runs before any LLM call; if triggered, the user is immediately told to contact emergency services, bypassing the LLM entirely
- **Graceful degradation** — if all configured models fail (rate limits, outages), the app returns a safe fallback message rather than crashing or hallucinating

---

## 📌 Known Limitations

- Uses **free-tier OpenRouter models**, which can occasionally return `429 Too Many Requests` during high demand. The app retries automatically and falls back to a secondary model, but very rarely may need a user retry.
- RAG knowledge is limited to whatever PDF is placed in `knowledge_base/` — it is not a general medical database.
- Not a substitute for professional medical advice under any circumstance.

---

## 📄 License

This project is for educational/portfolio purposes. Not intended for real-world clinical use.
