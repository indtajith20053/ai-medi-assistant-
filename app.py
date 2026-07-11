"""
Medical AI Assistant — Multi-Agent RAG System
================================================
Pure Python conversion of Medical_AI_Assistant_Correct.ipynb

Architecture: Supervisor + 4 specialist agents (symptom, medical-knowledge RAG,
medication, general) built on LangGraph, backed by a WHO-reference PDF indexed
in ChromaDB with MMR retrieval, served through a Gradio chat UI.

Usage:
    python app.py

Requires a `.env` file (same folder) or environment variable:
    OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx

Requires a WHO reference PDF at:
    knowledge_base/who_guideline.pdf
(relative to this script) the first time the vector store is built.
"""

try:
    # Must be imported first on Hugging Face ZeroGPU Spaces — before any
    # CUDA-touching package (torch/sentence-transformers) gets imported.
    # Safe to skip on CPU-only Spaces, Railway, Fly.io, or local runs.
    import spaces  # noqa: F401
except ImportError:
    pass

import os
import logging
import traceback
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END

import gradio as gr


# ---------------------------------------------------------------------------
# 1. Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("medical_assistant")


# ---------------------------------------------------------------------------
# 2. API Key Setup (.env)
# ---------------------------------------------------------------------------
load_dotenv()  # reads .env from the current working directory

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not found. Create a .env file in this script's "
        "folder with a line like:\n"
        "OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx\n"
        f"Current working directory: {os.getcwd()}"
    )

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

logger.info(
    f"API key configured (starts with: {OPENROUTER_API_KEY[:6]}..., "
    f"length: {len(OPENROUTER_API_KEY)})"
)


# ---------------------------------------------------------------------------
# 3. Model Configuration & LLM Factory (with Fallback)
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "supervisor": "meta-llama/llama-3.3-70b-instruct:free",
    "symptom": "openai/gpt-oss-120b:free",
    "medication": "openai/gpt-oss-120b:free",
    "rag": "nvidia/nemotron-3-ultra-550b-a55b:free",
}
FALLBACK_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

APP_REFERER = "https://github.com/your-username/medical-ai-assistant"
APP_TITLE_HEADER = "Medical AI Assistant"


def get_llm(model_name: str, temperature: float = 0.3) -> ChatOpenAI:
    """Build a LangChain ChatOpenAI client pointed at OpenRouter for the given model."""
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": APP_REFERER,
            "X-Title": APP_TITLE_HEADER,
        },
        timeout=30,
    )


def invoke_with_fallback(
    prompt: ChatPromptTemplate,
    inputs: dict,
    primary_model: str,
    temperature: float = 0.3,
    agent_name: str = "agent",
) -> str:
    """
    Invoke an LLM chain with automatic fallback to FALLBACK_MODEL if the primary model
    errors (rate-limited, deprecated, temporarily down, etc). Never raises — always
    returns a string, degrading gracefully on total failure.
    """
    for attempt, model_name in enumerate([primary_model, FALLBACK_MODEL], start=1):
        try:
            llm = get_llm(model_name, temperature=temperature)
            chain = prompt | llm
            response = chain.invoke(inputs)
            return getattr(response, "content", str(response))
        except Exception as e:
            logger.warning(f"[{agent_name}] Model '{model_name}' failed (attempt {attempt}): {e}")
            continue

    logger.error(f"[{agent_name}] All models failed.")
    return (
        "⚠️ I'm having trouble reaching the AI service right now. "
        "Please try again in a moment, and if the issue continues, consult a healthcare "
        "professional directly for anything urgent."
    )


# ---------------------------------------------------------------------------
# 4. Load WHO Reference PDF, Chunk, Embed, and Build/Load Vector Store
# ---------------------------------------------------------------------------
PDF_FILE = os.path.join("knowledge_base", "who_guideline.pdf")

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "who_collection"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def build_or_load_vectordb() -> Chroma:
    """
    Load an existing Chroma collection if one is already persisted at CHROMA_DIR.
    Otherwise, load the WHO reference PDF, chunk it, and build a fresh collection.
    """
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
    )

    try:
        existing_count = vectordb._collection.count()
    except Exception:
        existing_count = 0

    if existing_count > 0:
        logger.info(f"Loaded existing ChromaDB collection with {existing_count} chunk(s).")
        return vectordb

    if not os.path.exists(PDF_FILE):
        logger.warning(
            f"PDF not found at '{PDF_FILE}'. Starting with an empty vector store — "
            "RAG answers will fall back to '(No relevant reference material was found)' "
            "until a document is indexed."
        )
        return vectordb

    logger.info(f"Building vector store from '{PDF_FILE}'...")
    loader = PyPDFLoader(PDF_FILE)
    all_pages = loader.load()
    logger.info(f"Loaded {len(all_pages)} page(s).")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(all_pages)
    logger.info(f"Split {len(all_pages)} page(s) into {len(doc_chunks)} chunk(s).")

    vectordb.add_documents(doc_chunks)
    logger.info("✅ ChromaDB collection built and persisted.")
    return vectordb


vectordb = build_or_load_vectordb()


def retrieve_medical_context(query: str, k: int = 3, fetch_k: int = 10) -> str:
    """Retrieve the k most relevant + diverse chunks for a query using MMR."""
    try:
        results = vectordb.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
    except Exception as e:
        logger.error(f"[retrieve_medical_context] Chroma retrieval failed: {e}")
        return ""

    if not results:
        return ""

    excerpts = []
    for i, doc in enumerate(results, start=1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "?")
        excerpts.append(f"[Excerpt {i} — {source}, page {page}]\n{doc.page_content}")

    return "\n\n".join(excerpts)


# ---------------------------------------------------------------------------
# 5. Emergency Detection (Deterministic Safety Net)
# ---------------------------------------------------------------------------
EMERGENCY_TERMS = [
    "chest pain", "shortness of breath", "can't breathe", "cant breathe",
    "severe bleeding", "unconscious", "fainting", "stroke", "heart attack",
    "seizure", "overdose", "suicidal", "self-harm", "911",
    "call emergency", "emergency", "severe headache",
]

EMERGENCY_MESSAGE = (
    "🚑 This may be an emergency. Please call emergency services immediately "
    "or go to the nearest emergency department. Do not rely on this chat for urgent situations."
)


def contains_emergency_keywords(text: str) -> bool:
    """Deterministic keyword check — runs BEFORE any LLM call, so it can never be talked around."""
    if not text:
        return False
    lowered = text.lower()
    return any(term in lowered for term in EMERGENCY_TERMS)


# ---------------------------------------------------------------------------
# 6. Agent Prompts & Functions
# ---------------------------------------------------------------------------
SAFETY_RULES = """
Non-negotiable safety rules you must always follow:
- You are NOT a doctor and must never provide a medical diagnosis.
- Never tell the user what disease/condition they specifically have.
- Never recommend a medication dosage, or tell the user to start, stop, or change any medication.
- Always recommend that the user consult a licensed healthcare professional for diagnosis, treatment,
  or before making any medication decisions.
- If the user describes symptoms or a situation that could indicate a medical emergency, clearly and
  immediately advise them to contact emergency services or go to the nearest emergency room.
- Keep your tone calm, empathetic, and clear. Avoid overly technical jargon unless you explain it.
- Never claim certainty about something you don't have evidence for.
"""

# --- Symptom Assessment Agent ---
SYMPTOM_SYSTEM_PROMPT = f"""You are the Symptom Assessment Agent, part of a medical information assistant.

Your role:
- Help the user think through symptoms they are experiencing in an educational, supportive way.
- Ask at most 2-3 clarifying follow-up questions when needed (duration, severity, associated symptoms).
- Explain general categories of possible causes WITHOUT ever naming a single definitive diagnosis.
  Use phrasing like "this pattern can sometimes be associated with..." rather than "you have...".
- Always end by encouraging the user to see a doctor or nurse for proper assessment.
{SAFETY_RULES}
"""


def symptom_agent_respond(user_message: str, chat_history_text: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYMPTOM_SYSTEM_PROMPT),
        ("human", "Conversation so far:\n{history}\n\nUser message:\n{message}"),
    ])
    return invoke_with_fallback(
        prompt, {"history": chat_history_text, "message": user_message},
        primary_model=MODEL_CONFIG["symptom"], temperature=0.4, agent_name="symptom",
    )


# --- Medical Knowledge (RAG) Agent ---
RAG_SYSTEM_PROMPT = f"""You are the Medical Knowledge Agent, part of a medical information assistant.

Your role:
- Answer questions about diseases, conditions, treatments, and medical concepts using ONLY the
  reference context provided below, plus general knowledge to explain terms simply.
- If the context doesn't contain enough information, say so honestly rather than inventing an answer,
  and offer to answer using general knowledge instead (clearly labeled as general knowledge, not
  from the uploaded documents).
- Cite which excerpt(s) you used where relevant (e.g. "According to Excerpt 2...").
{SAFETY_RULES}

Reference context retrieved from the uploaded medical documents:
---
{{context}}
---
"""


def rag_agent_respond(user_message: str, chat_history_text: str) -> str:
    context = retrieve_medical_context(user_message)
    if not context:
        context = "(No relevant reference material was found for this query.)"
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "Conversation so far:\n{history}\n\nUser question:\n{message}"),
    ])
    return invoke_with_fallback(
        prompt, {"history": chat_history_text, "message": user_message, "context": context},
        primary_model=MODEL_CONFIG["rag"], temperature=0.3, agent_name="rag",
    )


# --- Medication Information Agent ---
MEDICATION_SYSTEM_PROMPT = f"""You are the Medication Information Agent, part of a medical information assistant.

Your role:
- Explain general information about medications: typical uses, common side effects, general
  precautions, storage, and general interaction categories — without personalized advice.
- NEVER state or imply a specific dosage or schedule.
- NEVER tell the user to start, stop, increase, decrease, or switch any medication.
- If asked for a dosage or "should I take/stop this", explain this must be decided by their doctor
  or pharmacist based on their personal medical history.
{SAFETY_RULES}
"""


def medication_agent_respond(user_message: str, chat_history_text: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", MEDICATION_SYSTEM_PROMPT),
        ("human", "Conversation so far:\n{history}\n\nUser question:\n{message}"),
    ])
    return invoke_with_fallback(
        prompt, {"history": chat_history_text, "message": user_message},
        primary_model=MODEL_CONFIG["medication"], temperature=0.3, agent_name="medication",
    )


# --- General / small-talk agent (no LLM call — saves free-tier quota) ---
def general_agent_respond(user_message: str) -> str:
    return (
        "Hi! 👋 I'm a medical information assistant. I can help you:\n"
        "- Think through symptoms you're experiencing (educational guidance only)\n"
        "- Explain diseases and medical concepts from the uploaded reference material\n"
        "- Give general information about medications\n\n"
        "I can't diagnose conditions or prescribe treatment — for that, please see a licensed "
        "healthcare professional. What would you like to ask?"
    )


# ---------------------------------------------------------------------------
# 7. Supervisor Routing Logic
# ---------------------------------------------------------------------------
VALID_ROUTES = {"symptom", "rag", "medication", "general"}

# Fast, deterministic baseline — catches obvious cases without spending an LLM call
KEYWORD_ROUTES = {
    "medication": ["medication", "medicine", "drug", "dose", "dosage", "tablet", "pill", "side effect"],
    "symptom": ["symptom", "feel sick", "pain", "ache", "fever", "nausea", "dizzy"],
}

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor Agent of a medical information assistant system.

Read the user's latest message and recent conversation context, then choose exactly ONE route:
- "symptom" — user is describing symptoms, feeling unwell, or asking what could be wrong.
- "rag" — user is asking to understand a disease, condition, medical concept, or procedure.
- "medication" — user is asking about a specific drug/medication (uses, side effects, storage, interactions).
- "general" — greeting, small talk, thanks, or anything unrelated to medical topics.

Respond with ONLY one lowercase word: symptom, rag, medication, or general. No explanation, no punctuation.
"""


def _keyword_prefilter(user_message: str) -> Optional[str]:
    lowered = user_message.lower()
    for route, terms in KEYWORD_ROUTES.items():
        if any(term in lowered for term in terms):
            return route
    return None


def supervisor_route(user_message: str, chat_history_text: str = "") -> str:
    """Decide which agent should answer: keyword pre-check first, then LLM classification."""
    keyword_hit = _keyword_prefilter(user_message)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_SYSTEM_PROMPT),
        ("human", "Conversation so far:\n{history}\n\nUser message:\n{message}"),
    ])
    raw_route = invoke_with_fallback(
        prompt, {"history": chat_history_text, "message": user_message},
        primary_model=MODEL_CONFIG["supervisor"], temperature=0.0, agent_name="supervisor",
    )
    route = raw_route.strip().lower()

    if route not in VALID_ROUTES:
        route = keyword_hit if keyword_hit else "general"
        logger.warning(f"[supervisor_route] Unrecognized LLM route '{raw_route}' — using '{route}'")

    return route


# ---------------------------------------------------------------------------
# 8. Build the LangGraph Multi-Agent Graph
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """Shared state passed between nodes in the graph."""
    user_message: str
    chat_history_text: str
    route: str
    final_response: str


def supervisor_node(state: AgentState) -> AgentState:
    """Decide which specialist agent should handle this message (validated)."""
    raw_route = supervisor_route(state["user_message"], state["chat_history_text"])
    route = str(raw_route).strip().lower()
    if route not in VALID_ROUTES:
        logger.warning(f"[supervisor_node] Unexpected route '{raw_route}' — defaulting to 'general'")
        route = "general"
    logger.info(f"Routing decision: {route}")  # log the decision, not the message content
    return {**state, "route": route}


def symptom_node(state: AgentState) -> AgentState:
    answer = symptom_agent_respond(state["user_message"], state["chat_history_text"])
    return {**state, "final_response": answer}


def rag_node(state: AgentState) -> AgentState:
    answer = rag_agent_respond(state["user_message"], state["chat_history_text"])
    return {**state, "final_response": answer}


def medication_node(state: AgentState) -> AgentState:
    answer = medication_agent_respond(state["user_message"], state["chat_history_text"])
    return {**state, "final_response": answer}


def general_node(state: AgentState) -> AgentState:
    answer = general_agent_respond(state["user_message"])
    return {**state, "final_response": answer}


def route_decision(state: AgentState) -> str:
    return state["route"]


graph_builder = StateGraph(AgentState)

graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("symptom", symptom_node)
graph_builder.add_node("rag", rag_node)
graph_builder.add_node("medication", medication_node)
graph_builder.add_node("general", general_node)

graph_builder.set_entry_point("supervisor")

graph_builder.add_conditional_edges(
    "supervisor",
    route_decision,
    {
        "symptom": "symptom",
        "rag": "rag",
        "medication": "medication",
        "general": "general",
    },
)

graph_builder.add_edge("symptom", END)
graph_builder.add_edge("rag", END)
graph_builder.add_edge("medication", END)
graph_builder.add_edge("general", END)

medical_assistant_graph = graph_builder.compile()


# ---------------------------------------------------------------------------
# 9. Top-Level Entry Point
# ---------------------------------------------------------------------------
def run_medical_assistant(user_message: str, chat_history_text: str = "") -> str:
    """
    Top-level entry point used by the UI.
    1. Checks for emergency keywords FIRST (deterministic safety net, runs before any LLM call).
    2. Otherwise, runs the LangGraph supervisor + specialist pipeline.
    """
    if contains_emergency_keywords(user_message):
        logger.info("Emergency keywords detected — short-circuiting to emergency response.")
        return EMERGENCY_MESSAGE

    try:
        result = medical_assistant_graph.invoke({
            "user_message": user_message,
            "chat_history_text": chat_history_text,
            "route": "",
            "final_response": "",
        })
        return result["final_response"]
    except Exception as e:
        logger.error(f"[run_medical_assistant] Unhandled error: {e}")
        traceback.print_exc()
        return (
            "⚠️ Sorry, something went wrong while processing your request. "
            "Please try again, and if the issue continues, consult a healthcare "
            "professional directly for anything urgent."
        )


# ---------------------------------------------------------------------------
# 10. Gradio UI
# ---------------------------------------------------------------------------
APP_TITLE = "🩺 Medical AI Assistant"

DISCLAIMER = (
    "**Disclaimer:** This assistant provides general educational information only. "
    "It does not diagnose conditions or prescribe treatment. "
    "Always consult a qualified healthcare provider."
)

EXAMPLE_QUESTIONS = [
    "What is hypertension?",
    "What are symptoms of diabetes?",
    "What are common side effects of ibuprofen?",
]


def stream_response(user_message, history):
    if not user_message or not str(user_message).strip():
        yield "Please type a question."
        return

    chat_history_text = ""
    if history:
        messages = []
        for h in history:
            if isinstance(h, dict):
                messages.append(f"{h['role'].capitalize()}: {h['content']}")
            elif isinstance(h, (list, tuple)) and len(h) >= 2:
                messages.append(f"User: {h[0]}\nAssistant: {h[1]}")
        chat_history_text = "\n".join(messages)

    answer = run_medical_assistant(user_message, chat_history_text)
    yield answer


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(DISCLAIMER)

        chatbot = gr.Chatbot(label="Medical AI Assistant", height=480)

        gr.ChatInterface(
            fn=stream_response,
            chatbot=chatbot,
            examples=EXAMPLE_QUESTIONS,
        )

    return demo


def main():
    logger.info("Running self-test...")
    logger.info(run_medical_assistant("Hi there!"))

    demo = build_ui()
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
