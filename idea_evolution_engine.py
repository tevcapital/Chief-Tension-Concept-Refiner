import re
import sys

from llm import LLMConfig, call_llm


# ─── CONFIG ───────────────────────────────────────────────────────────────────

BACKEND = "ollama"          # "groq" or "ollama"

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = "your-groq-api-key-here"

OLLAMA_MODEL = "qwen3.5:4b"
OLLAMA_URL = "http://localhost:11434"

# ──────────────────────────────────────────────────────────────────────────────

CONCEPT_PROMPT = """Generate 16 DISTINCT ideas for the following domain: {domain}

Rules:
- Each idea must come from a different angle (business model, arbitrage, service, product, distribution, etc.)
- Avoid repeating patterns
- Be bold and non-obvious
- Keep each idea concise but clear

Output exactly as:
IDEA 1: ...
IDEA 2: ...
...
IDEA 16: ..."""

TENSION_PROMPT = """You are stress-testing ideas.{depth_note}

For EACH idea:
- Identify the main weakness
- Identify what must be true for it to work
- Identify the biggest real-world risk

IMPORTANT:
- Do NOT repeat generic criticism
- Be specific and practical

Ideas:
{ideas_text}

Output format:

IDEA 1:
WEAKNESS: ...
ASSUMPTION: ...
RISK: ...

IDEA 2:
WEAKNESS: ...
ASSUMPTION: ...
RISK: ...

(continue for all ideas)"""

CHIEF_PROMPT = """You are selecting the strongest ideas.

Ideas:
{ideas_text}

Critiques:
{critiques_text}

Select the BEST {target_n} ideas.

RULES:
- Be aggressive in eliminating weak ideas
- Use critiques heavily in your decision
- Favor ideas that survived criticism well
- Do NOT be fair — only keep the strongest

CRITERIA:
- speed to make money
- clarity of value
- execution feasibility
- resilience to criticism

OUTPUT:

SELECTED:
- IDEA X: reason
- IDEA Y: reason
(select exactly {target_n} ideas)"""

REFINE_PROMPT = """Improve the following idea using the critique:

IDEA:
{idea}

CRITIQUE:
{critique}

Make it:
- more practical
- more executable
- more monetizable
- stronger against the identified weaknesses

Do NOT just rephrase — improve it."""

DEEP_REFINE_PROMPT = """Push this idea further. Make it sharper, more practical, and more monetizable.

IDEA:
{idea}

- Remove any remaining vagueness
- Make the first $1 path clearer
- Tighten the execution steps

Do NOT just rephrase — make it meaningfully better."""


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def log(step: str, content: str = ""):
    print(f"\n{'─' * 60}")
    print(f"  EVOLUTION ENGINE | {step}")
    print(f"{'─' * 60}")
    if content:
        print(content)


def ideas_to_text(ideas: list) -> str:
    return "\n\n".join(f"IDEA {i+1}: {idea}" for i, idea in enumerate(ideas))


def critiques_to_text(critiques: list) -> str:
    return "\n\n".join(f"IDEA {i+1} CRITIQUE:\n{c}" for i, c in enumerate(critiques))


def parse_ideas(text: str, count: int) -> list:
    """Parse IDEA N: ... blocks into a list of content strings."""
    ideas = []
    for i in range(1, count + 1):
        next_marker = rf"IDEA {i+1}:" if i < count else "$"
        pattern = rf"IDEA {i}:\s*(.*?)(?={next_marker})"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        ideas.append(match.group(1).strip() if match else f"[idea {i} not parsed]")
    return ideas


def parse_critiques(text: str, count: int) -> list:
    """Parse IDEA N: critique blocks into a list of strings."""
    critiques = []
    for i in range(1, count + 1):
        next_marker = rf"IDEA {i+1}:" if i < count else "$"
        pattern = rf"IDEA {i}:\s*(.*?)(?={next_marker})"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        critiques.append(match.group(1).strip() if match else "[no critique]")
    return critiques


def parse_selected(text: str, ideas: list, critiques: list, target_n: int):
    """Extract selected idea indices and return (selected_ideas, selected_critiques)."""
    numbers = re.findall(r"IDEA\s+(\d+)", text, re.IGNORECASE)
    seen = set()
    indices = []
    for n in numbers:
        idx = int(n) - 1
        if 0 <= idx < len(ideas) and idx not in seen:
            seen.add(idx)
            indices.append(idx)

    # fallback: take first target_n if not enough parsed
    if len(indices) < target_n:
        for i in range(len(ideas)):
            if i not in seen:
                indices.append(i)
                seen.add(i)
            if len(indices) >= target_n:
                break

    indices = indices[:target_n]
    return [ideas[i] for i in indices], [critiques[i] for i in indices]


# ─── AGENTS ───────────────────────────────────────────────────────────────────

def generate_ideas(domain: str, config: LLMConfig) -> list:
    log("CONCEPT AGENT", f"Generating 16 ideas for: {domain}...")
    system = "You are a world-class creative strategist. Generate bold, distinct, non-obvious ideas."
    output = call_llm(config, system, CONCEPT_PROMPT.format(domain=domain))
    log("CONCEPT AGENT → Output", output)
    ideas = parse_ideas(output, 16)
    return ideas


def tension_analyze(ideas: list, config: LLMConfig, round_num: int = 1) -> list:
    depth_note = ""
    if round_num > 1:
        depth_note = f"\nIMPORTANT: This is Round {round_num}. Go DEEPER and MORE CRITICAL — these ideas already survived earlier rounds."
    prompt = TENSION_PROMPT.format(
        depth_note=depth_note,
        ideas_text=ideas_to_text(ideas),
    )
    log(f"TENSION AGENT | Round {round_num}", f"Critiquing {len(ideas)} ideas...")
    system = "You are a ruthless product critic. Identify real weaknesses, false assumptions, and execution risks."
    output = call_llm(config, system, prompt)
    log("TENSION AGENT → Output", output)
    return parse_critiques(output, len(ideas))


def chief_select(ideas: list, critiques: list, target_n: int, config: LLMConfig, round_num: int = 1):
    prompt = CHIEF_PROMPT.format(
        ideas_text=ideas_to_text(ideas),
        critiques_text=critiques_to_text(critiques),
        target_n=target_n,
    )
    log(f"CHIEF | Round {round_num}", f"Selecting best {target_n} from {len(ideas)} ideas...")
    system = "You are a chief product officer. Select only the strongest ideas. Be ruthless and decisive."
    output = call_llm(config, system, prompt)
    log("CHIEF → Selection", output)
    return parse_selected(output, ideas, critiques, target_n)


def refine_batch(ideas: list, critiques: list, config: LLMConfig, round_num: int = 1) -> list:
    refined = []
    system = "You are a product refiner. Make ideas more practical, executable, and monetizable."
    for i, (idea, critique) in enumerate(zip(ideas, critiques)):
        log(f"REFINER | Round {round_num} | {i+1}/{len(ideas)}", "")
        prompt = REFINE_PROMPT.format(idea=idea, critique=critique[:600])
        output = call_llm(config, system, prompt)
        log(f"REFINER → Idea {i+1}", output)
        refined.append(output.strip())
    return refined


def refine_idea(idea: str, config: LLMConfig, rounds: int = 4) -> str:
    current = idea
    system = "You are a product refiner doing deep iteration. Each pass must make the idea meaningfully better."
    for i in range(rounds):
        log(f"FINAL REFINER | Pass {i+1}/{rounds}", "")
        current = call_llm(config, system, DEEP_REFINE_PROMPT.format(idea=current))
        log(f"FINAL REFINER → Pass {i+1}", current)
    return current


# ─── PIPELINE ─────────────────────────────────────────────────────────────────

def run_pipeline(domain: str):
    config = LLMConfig(
        backend=BACKEND,
        model=OLLAMA_MODEL if BACKEND == "ollama" else GROQ_MODEL,
        api_key=GROQ_API_KEY if BACKEND == "groq" else "",
        ollama_url=OLLAMA_URL,
    )

    log("PIPELINE START", f"Domain: {domain}")

    # Step 1: Generate 16 ideas
    ideas = generate_ideas(domain, config)

    # Step 2: Multi-round elimination 16 → 8 → 4 → 2 → 1
    round_num = 1
    for target in [8, 4, 2, 1]:
        log(f"ROUND {round_num}", f"{len(ideas)} ideas → selecting {target}")
        critiques = tension_analyze(ideas, config, round_num)
        ideas, critiques = chief_select(ideas, critiques, target, config, round_num)
        if target > 1:
            ideas = refine_batch(ideas, critiques, config, round_num)
        round_num += 1

    # Step 3: Deep refinement of the winner (4 passes)
    log("FINAL ROUND", "Deep refining the winning idea (4 passes)...")
    final = refine_idea(ideas[0], config, rounds=4)

    log("FINAL RESULT", final)
    return final


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python3 idea_evolution_engine.py "your domain here"')
        sys.exit(1)
    domain = " ".join(sys.argv[1:])
    run_pipeline(domain)
