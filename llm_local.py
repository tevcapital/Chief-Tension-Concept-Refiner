import subprocess


def call_ollama(prompt: str, model: str = "qwen3.5:4b", temperature: float = 0.7) -> str:
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise Exception(result.stderr)
    return result.stdout.strip()
