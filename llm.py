# llm.py

# This file handles:
# - Sending query + context to LLM
# - Getting final answer

import os
from openai import OpenAI

# Add your API key securely
api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
client = OpenAI(api_key=api_key)


def generate_answer(query, context):
    """Generate answer using LLM based on retrieved context"""

    prompt = f"""
    You are a helpful customer support assistant.

    Answer the question ONLY using the context below.
    If answer is not in context, say "I don't know".

    Context:
    {context}

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content