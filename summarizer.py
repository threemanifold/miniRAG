from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

# Make sure your OpenAI API key is set:
# export OPENAI_API_KEY="your_api_key_here"

# Initialize the model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

def get_summarizer_model_name() -> str:
    try:
        return getattr(llm, "model_name", getattr(llm, "model", "unknown"))
    except Exception:
        return "unknown"

# Prompt template for summarizing a retrieved chunk
summary_prompt = PromptTemplate(
    input_variables=["text", "chunk"],
    template="""
You are an assistant helping to summarize retrieved context for a question.

Question:
{text}

Retrieved context:
{chunk}

Provide a concise 2-3 sentence summary of the retrieved context relevant to the question.
If the context is irrelevant, state: "No relevant information found."
"""
)

def batch_summarize_chunks(question: str, chunks: list[str]) -> list[str]:
    """Batch summarize multiple chunks in parallel using LangChain's invoke batching."""
    prompts = [summary_prompt.format(text=question, chunk=chunk) for chunk in chunks]
    responses = llm.batch(prompts)  # runs calls in parallel under the hood
    return [res.content.strip() for res in responses]


if __name__ == "__main__":
    # Example usage:
    question = "What are transformers and how do they work?"
    chunk = "Transformers are a type of deep learning architecture that rely on self-attention..."
    print(batch_summarize_chunks(question, [chunk]))
