from dataclasses import dataclass, field
from typing import Literal

import requests


@dataclass
class LLMConfig:
    backend: Literal["groq", "ollama"]
    model: str
    api_key: str = ""
    ollama_url: str = "http://localhost:11434"


def call_llm(config: LLMConfig, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    if config.backend == "groq":
        from groq import Groq
        client = Groq(api_key=config.api_key)
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
        )
        return response.choices[0].message.content

    elif config.backend == "ollama":
        response = requests.post(
            f"{config.ollama_url}/api/chat",
            json={"model": config.model, "messages": messages, "stream": False, "think": False},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    else:
        raise ValueError(f"Unknown backend: {config.backend}")
