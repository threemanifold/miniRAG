from typing import List, Tuple, Dict, Any, Iterable
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

SYSTEM_MSG = """You are a careful assistant.
You may use prior conversation turns for context and disambiguation.
However, all factual claims MUST be grounded in the provided chunks and cited as: (chunk index: <INDEX>).
If the chunks do not contain the needed info, say you don't have enough information.
Be concise and precise."""

HUMAN_TEMPLATE = """Question:
{question}

You are given {n_chunks} chunks. Each chunk is labeled by its original chunk_index.

Chunks:
{chunks_block}

Instructions:
- Try to understand the intent of the question, even if it's not clear.
- Use only information from the chunks for factual claims.
- Cite each claim with (chunk index: <INDEX>).
- If combining multiple chunks, cite like (chunk index: 0, 3).
- If nothing is relevant, say: "I don't have enough information in the provided chunks."
Provide your final answer now.
"""

def get_agent_prompt_text() -> str:
    return f"SYSTEM:\n{SYSTEM_MSG}\n\nHUMAN TEMPLATE:\n{HUMAN_TEMPLATE}"

def _format_chunks(retrieved: List[Tuple[str, Dict[str, Any]]]) -> str:
    lines = []
    for text, meta in retrieved:
        idx = meta.get("chunk_index", "NA")
        src = meta.get("source", "unknown")
        lines.append(f"[chunk_index={idx}] (source={src})\n{text}")
    return "\n\n---\n\n".join(lines)

_llm = ChatOpenAI(model="gpt-4o", temperature=0)

def get_answer_model_name() -> str:
    try:
        # ChatOpenAI exposes model name via .model or .model_name depending on version
        return getattr(_llm, "model_name", getattr(_llm, "model", "unknown"))
    except Exception:
        return "unknown"

def answer_with_citations(
    question: str,
    retrieved: List[Tuple[str, Dict[str, Any]]],
    history: Iterable[Tuple[str, str]] | None = None,
    max_history_turns: int = 6,
) -> str:
    if not retrieved:
        return "I don't have enough information in the provided chunks."

    messages = [SystemMessage(content=SYSTEM_MSG)]

    # Append recent history as proper message objects
    if history:
        trimmed = list(history)[-2 * max_history_turns :]
        for role, content in trimmed:
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                messages.append(AIMessage(content=content))

    # Current turn (human) with formatted template
    human_content = HUMAN_TEMPLATE.format(
        question=question,
        n_chunks=len(retrieved),
        chunks_block=_format_chunks(retrieved),
    )
    messages.append(HumanMessage(content=human_content))

    resp = _llm.invoke(messages)
    return resp.content.strip()


if __name__ == "__main__":
    # Tiny demo
    sample_chunks = [
        (
            "Transformers are a type of deep learning architecture introduced in 2017 with self-attention.",
            {"chunk_index": 0, "source": "transformers.txt"}
        ),
        (
            "Self-attention lets the model weigh tokens against each other to capture long-range dependencies.",
            {"chunk_index": 1, "source": "transformers.txt"}
        ),
    ]
    q = "What is a transformer and why is self-attention important?"
    print(answer_with_citations(q, sample_chunks))