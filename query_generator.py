from __future__ import annotations

from typing import Iterable, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


QUERY_SYSTEM_MSG = (
    "You are a search query rewriter for a RAG system. "
    "Rewrite the user's input into a concise, retrieval-friendly query using the user's words. "
    "Prefer specific nouns and key phrases; drop filler. "
    "Do NOT answer the question. Return ONLY the rewritten query as plain text."
)

QUERY_TEMPLATE = (
    "Original question:\n{question}\n\n"
    "Guidelines:\n"
    "- Preserve core entities and terms.\n"
    "- Expand abbreviations if obvious.\n"
    "- Remove politeness and extraneous words.\n"
    "- Keep it under 20 words.\n\n"
    "Return only the rewritten query."
)

_q_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_query_model_name() -> str:
    try:
        return getattr(_q_llm, "model_name", getattr(_q_llm, "model", "unknown"))
    except Exception:
        return "unknown"


def rewrite_query(
    question: str,
    history: Iterable[Tuple[str, str]] | None = None,
    max_history_turns: int = 4,
) -> str:
    messages = [SystemMessage(content=QUERY_SYSTEM_MSG)]

    # Include brief recent history if provided, to aid disambiguation
    if history:
        trimmed = list(history)[-2 * max_history_turns :]
        for role, content in trimmed:
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=QUERY_TEMPLATE.format(question=question)))

    resp = _q_llm.invoke(messages)
    return resp.content.strip()


