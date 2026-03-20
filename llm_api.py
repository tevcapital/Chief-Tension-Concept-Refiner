import os
from openai import OpenAI


def call_api(prompt: str, model: str, temperature: float = 0.3) -> str:
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("API_BASE_URL"),
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()
