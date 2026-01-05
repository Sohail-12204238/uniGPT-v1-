import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tavily import TavilyClient

from .config import FALLBACK_PHRASE
from .retrieval import get_retriever

# ---------------------------------------------------
# Prompt
# ---------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    """You are an assistant that answers students' questions about university details, rules, and guidelines.
Always use the provided context to answer.

If the answer is found in the context:
- Respond clearly in 3–5 sentences.
- Do not add extra information.

If the context does NOT contain the answer, say:
"I'm sorry, I could not find that information in the university guidelines."

<context>
{context}
</context>

Question: {input}
"""
)

# ---------------------------------------------------
# Lazy clients
# ---------------------------------------------------
_llm = None
_tavily_client = None


def get_llm():
    global _llm
    if _llm is not None:
        return _llm

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    _llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=500,
    )
    return _llm


def get_tavily_client():
    global _tavily_client

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None

    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=api_key)

    return _tavily_client


def web_search_fallback(question: str):
    client = get_tavily_client()
    if not client:
        return "Web search is not available.", []

    try:
        resp = client.search(
            query=question,
            search_depth="basic",
            max_results=5,
            include_answer=True,
        )
    except Exception as e:
        return f"Web search error: {e}", []

    answer = resp.get("answer", "No clear answer found.")
    sources = [
        r.get("url")
        for r in resp.get("results", [])
        if isinstance(r, dict) and r.get("url")
    ]

    return answer, sources


def answer_question(question: str) -> dict:
    retriever = get_retriever()
    docs = retriever.invoke(question)

    context = "\n\n".join(d.page_content for d in docs) if docs else ""

    pdf_sources = [
        f"{d.metadata.get('source')} (page {d.metadata.get('page', '?')})"
        for d in docs
    ]

    llm = get_llm()
    formatted = prompt.invoke({"context": context, "input": question})
    result = llm.invoke(formatted)
    answer = result.content

    from_web = False
    web_sources = []

    if not docs or (FALLBACK_PHRASE and FALLBACK_PHRASE in answer):
        from_web = True
        web_answer, web_sources = web_search_fallback(question)
        answer = f"{web_answer}\n\n(Note: This answer is from web search.)"

    return {
        "answer": answer,
        "pdf_sources": pdf_sources,
        "web_sources": web_sources,
        "from_web": from_web,
    }
