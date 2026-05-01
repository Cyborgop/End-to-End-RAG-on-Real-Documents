import os
from openai import OpenAI

_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
        _client = OpenAI(api_key=api_key)
    return _client

def generate_answer(query, context_chunks, model="gpt-3.5-turbo"):
    context = "\n\n".join(context_chunks)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question using only "
                "the provided context. If the context does not contain enough "
                "information to answer, say so explicitly."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]
    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content
