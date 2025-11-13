"""
generator.py
-------------
Minimal LLM answer generator that formats retrieved contexts and
asks the model to answer concisely with citations.
"""

import os
from typing import List, Dict
from openai import OpenAI


class Generator:
    """
    Simple wrapper around OpenAI Chat Completions for RAG answering.
    """

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")

    def answer(self, question: str, contexts: List[Dict]) -> str:
        """
        Build a compact prompt with retrieved contexts and ask the LLM.

        Notes
        -----
        - We keep the prompt short to minimize token usage.
        - Each context is labeled [Ci] to allow lightweight "citations".
        """
        context_blocks = []
        for i, c in enumerate(contexts, start=1):
            src = c.get("metadata", {}).get("source", "unknown")
            snippet = c.get("text", "")[:800]
            context_blocks.append(f"[C{i}] Source: {src}\n{snippet}")

        prompt = (
            "You are a helpful assistant. Use the provided contexts to answer the user's question.\n"
            "Cite evidence by referencing [C1], [C2], ... when relevant. "
            "If you are unsure or context is insufficient, say so briefly.\n\n"
            f"Question: {question}\n\n"
            "Contexts:\n" + "\n\n".join(context_blocks)
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Be concise, accurate, and cite [C#] when applicable."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
