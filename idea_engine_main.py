import re
import sys

from llm import LLMConfig
from idea_engine_agents import concept_agent, tension_agent, chief_pick_agent, refiner_agent


# ─── CONFIG ───────────────────────────────────────────────────────────────────

BACKEND = "ollama"          # "groq" or "ollama"

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = "your-groq-api-key-here"

OLLAMA_MODEL = "qwen3.5:4b"
OLLAMA_URL = "http://localhost:11434"

# ──────────────────────────────────────────────────────────────────────────────


def log(step: str, content: str = ""):
    print(f"\n{'─' * 60}")
    print(f"  IDEA ENGINE | {step}")
    print(f"{'─' * 60}")
    if content:
        print(content)


def extract_block(text: str, label: str, number: int) -> str:
    """Extract a numbered block (e.g. IDEA 2 or CRITIQUE 3) from agent output."""
    pattern = rf"{label} {number}:.*?(?={label} {number + 1}:|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(0).strip() if match else text


def parse_pick(pick_text: str) -> int:
    """Parse the idea number from the chief's pick response."""
    match = re.search(r"PICK:\s*([123])", pick_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # fallback: find any standalone 1-3
    match = re.search(r"\b([123])\b", pick_text)
    return int(match.group(1)) if match else 1


def run(domain: str):
    config = LLMConfig(
        backend=BACKEND,
        model=OLLAMA_MODEL if BACKEND == "ollama" else GROQ_MODEL,
        api_key=GROQ_API_KEY if BACKEND == "groq" else "",
        ollama_url=OLLAMA_URL,
    )

    log("CHIEF", f"Domain: {domain}")

    # Step 1: Generate 3 ideas
    log("CONCEPT AGENT", f"Generating 3 ideas for: {domain}...")
    ideas = concept_agent(domain, config)
    log("CONCEPT AGENT → Output", ideas)

    # Step 2: Critique all 3
    log("TENSION AGENT", "Critiquing all 3 ideas...")
    critiques = tension_agent(ideas, config)
    log("TENSION AGENT → Output", critiques)

    # Step 3: Chief picks the best
    log("CHIEF", "Selecting best idea...")
    pick_response = chief_pick_agent(ideas, critiques, config)
    log("CHIEF → Selection", pick_response)
    pick_number = parse_pick(pick_response)
    print(f"\n  → Chose Idea #{pick_number}")

    # Step 4: Extract chosen idea + its critique
    chosen_idea = extract_block(ideas, "IDEA", pick_number)
    chosen_critique = extract_block(critiques, "CRITIQUE", pick_number)[:800]

    # Step 5: Refine
    log("REFINER AGENT", f"Refining Idea #{pick_number}...")
    refined = refiner_agent(chosen_idea, chosen_critique, config)
    log("REFINER AGENT → Output", refined)

    log("CHIEF → FINAL OUTPUT", refined)
    return refined


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python3 idea_engine_main.py "domain"')
        print('Example: python3 idea_engine_main.py healthcare')
        print('Example: python3 idea_engine_main.py "solo entrepreneurs"')
        sys.exit(1)
    domain = " ".join(sys.argv[1:])
    run(domain)
