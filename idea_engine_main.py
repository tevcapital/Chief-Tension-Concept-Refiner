import re
import sys

from llm import LLMConfig, call_llm
from idea_engine_agents import tension_agent, refiner_agent


# ─── CONFIG ───────────────────────────────────────────────────────────────────

BACKEND = "ollama"          # "groq" or "ollama"

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = "your-groq-api-key-here"

OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_URL = "http://localhost:11434"

# ──────────────────────────────────────────────────────────────────────────────

CONCEPT_SYSTEM = """You are a world-class strategist. Generate exactly 6 distinct, high-quality ideas for the given domain.

Strict rules:
- Each idea must represent a fundamentally different strategy
- If two ideas are similar or variations of each other, merge them into one stronger idea
- No filler ideas — every idea must earn its place
- Be specific: who does it, what happens, why it works
- Each idea must have a clear competitive advantage or measurable upside

Format exactly as:

IDEA 1: [title]
[2-3 sentences]

IDEA 2: [title]
[2-3 sentences]

IDEA 3: [title]
[2-3 sentences]

IDEA 4: [title]
[2-3 sentences]

IDEA 5: [title]
[2-3 sentences]

IDEA 6: [title]
[2-3 sentences]"""

CHIEF_SYSTEM = """You are a chief product officer. You select the strongest ideas based on real-world potential.

Criteria:
- speed to value
- clarity of execution
- resilience to criticism
- differentiation

Be aggressive. Eliminate weak ideas. Justify every selection.

Respond ONLY in this format:

SELECTED:
- IDEA [number]: [one sentence justification]
- IDEA [number]: [one sentence justification]
(repeat for exactly {target_n} selections)"""


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def log(step: str, content: str = ""):
    print(f"\n{'─' * 60}")
    print(f"  IDEA ENGINE | {step}")
    print(f"{'─' * 60}")
    if content:
        print(content)


def format_ideas(ideas: list) -> str:
    return "\n\n".join(f"IDEA {i+1}: {idea}" for i, idea in enumerate(ideas))


def parse_ideas(text: str, count: int) -> list:
    ideas = []
    for i in range(1, count + 1):
        next_marker = rf"IDEA {i+1}:" if i < count else "$"
        pattern = rf"IDEA {i}:\s*(.*?)(?={next_marker})"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        ideas.append(match.group(1).strip() if match else f"[idea {i} not parsed]")
    return ideas


def extract_block(text: str, label: str, number: int) -> str:
    pattern = rf"{label} {number}:.*?(?={label} {number + 1}:|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(0).strip() if match else ""


def parse_chief_selection(text: str, ideas: list, critiques_text: str, target_n: int) -> tuple:
    """Return (selected_ideas, selected_indices) based on chief output."""
    numbers = re.findall(r"IDEA\s+(\d+)", text, re.IGNORECASE)
    seen = set()
    indices = []
    for n in numbers:
        idx = int(n) - 1
        if 0 <= idx < len(ideas) and idx not in seen:
            seen.add(idx)
            indices.append(idx)

    # fallback: take first target_n
    if len(indices) < target_n:
        for i in range(len(ideas)):
            if i not in seen:
                indices.append(i)
                seen.add(i)
            if len(indices) >= target_n:
                break

    return [ideas[i] for i in indices[:target_n]], indices[:target_n]


# ─── AGENTS ───────────────────────────────────────────────────────────────────

def generate_ideas(domain: str, config: LLMConfig) -> list:
    log("CONCEPT AGENT", f"Generating 6 distinct ideas for: {domain}...")
    output = call_llm(config, CONCEPT_SYSTEM, f"Domain: {domain}")
    log("CONCEPT AGENT → Output", output)
    return parse_ideas(output, 6)


def chief_select(ideas: list, critiques_text: str, target_n: int, config: LLMConfig, round_num: int) -> tuple:
    system = CHIEF_SYSTEM.format(target_n=target_n)
    prompt = f"IDEAS:\n{format_ideas(ideas)}\n\nCRITIQUES:\n{critiques_text}\n\nSelect the best {target_n} ideas."
    log(f"CHIEF | Round {round_num}", f"Selecting top {target_n} from {len(ideas)} ideas...")
    output = call_llm(config, system, prompt)
    log(f"CHIEF → Selection", output)
    return parse_chief_selection(output, ideas, critiques_text, target_n)


def compress_to_decision(text: str, config: LLMConfig) -> str:
    system = "You are a decision compressor. Convert verbose ideas into clear, concrete, executable actions."
    prompt = f"""Rewrite the following idea into a clear, concrete, executable action.

RULES:
- Maximum 2 sentences
- Must describe a real-world action (not a concept)
- No branding names (no "Engine", "Protocol", "Framework", etc.)
- No metaphors or storytelling
- No vague language (no "leverage", "optimize", "reimagine", "enhance")
- Must clearly state what is being done, for whom, and how it creates value

IDEA:
{text}

OUTPUT:"""
    return call_llm(config, system, prompt).strip()


def refine_ideas(ideas: list, critiques_text: str, config: LLMConfig, round_num: int) -> list:
    refined = []
    for i, idea in enumerate(ideas):
        critique = extract_block(critiques_text, "CRITIQUE", i + 1) or critiques_text[:600]
        log(f"REFINER | Round {round_num} | {i+1}/{len(ideas)}", "")
        output = refiner_agent(idea, critique[:600], config)
        log(f"REFINER → Idea {i+1}", output)
        refined.append(output.strip())
    return refined


# ─── PIPELINE ─────────────────────────────────────────────────────────────────

def run(domain: str):
    config = LLMConfig(
        backend=BACKEND,
        model=OLLAMA_MODEL if BACKEND == "ollama" else GROQ_MODEL,
        api_key=GROQ_API_KEY if BACKEND == "groq" else "",
        ollama_url=OLLAMA_URL,
    )

    log("CHIEF", f"Domain: {domain}")

    # ── STAGE 1: Generate 6 ideas ──────────────────────────────────────────────
    ideas = generate_ideas(domain, config)

    # ── STAGE 2: Tension Round 1 — critique all 6 ─────────────────────────────
    log("TENSION AGENT | Round 1", "Critiquing all 6 ideas...")
    critiques_1 = tension_agent(format_ideas(ideas), config)
    log("TENSION AGENT → Round 1 Output", critiques_1)

    # ── STAGE 3: Chief Round 1 — select top 3 ─────────────────────────────────
    ideas_3, _ = chief_select(ideas, critiques_1, 3, config, round_num=1)

    # ── STAGE 4: Refiner Round 1 — refine the 3 ───────────────────────────────
    ideas_3 = refine_ideas(ideas_3, critiques_1, config, round_num=1)

    # ── STAGE 5: Tension Round 2 — deeper critique of 3 ──────────────────────
    log("TENSION AGENT | Round 2", "Deeper critique of 3 refined ideas...")
    critiques_2 = tension_agent(format_ideas(ideas_3), config)
    log("TENSION AGENT → Round 2 Output", critiques_2)

    # ── STAGE 6: Chief Round 2 — select top 1 directly from 3 ────────────────
    log("CHIEF | Round 2", "Selecting 1 from 3 ideas...")
    final_ideas, _ = chief_select(ideas_3, critiques_2, 1, config, round_num=2)
    final_idea = final_ideas[0]

    # ── STAGE 7: Refiner Round 2 — refine the winner ─────────────────────────
    final_ideas = refine_ideas([final_idea], critiques_2, config, round_num=2)
    final_idea = final_ideas[0]

    # ── STAGE 8: Compress to primary decision ─────────────────────────────────
    log("COMPRESSION AGENT", "Compressing to primary decision...")
    decision = compress_to_decision(final_idea, config)

    # ── STAGE 9: Final Output ──────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  PRIMARY DECISION")
    print(f"{'═' * 60}")
    print(decision)

    return decision


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python3 idea_engine_main.py "domain"')
        print('Example: python3 idea_engine_main.py healthcare')
        print('Example: python3 idea_engine_main.py "solo entrepreneurs"')
        sys.exit(1)
    domain = " ".join(sys.argv[1:])
    run(domain)
