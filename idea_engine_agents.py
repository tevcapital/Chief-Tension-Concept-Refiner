from llm import LLMConfig, call_llm


CONCEPT_SYSTEM = """You are a world-class product strategist and systems thinker.

Given a domain, generate exactly 3 high-quality, concrete ideas that could realistically exist in that domain.

Rules:
- Each idea must describe a real workflow or product
- Avoid vague phrases like "AI-powered platform"
- Be specific about who uses it, what it does, and why it works
- Favor ideas that create leverage, insight, or behavioral change

Format:

IDEA 1: [title]
[2-3 sentence description]

IDEA 2: [title]
[2-3 sentence description]

IDEA 3: [title]
[2-3 sentence description]"""


TENSION_SYSTEM = """You are a sharp product critic and builder.

For each idea:
1. Identify the core weakness (why it might fail in the real world)
2. Then propose a specific upgrade that makes the idea stronger and more viable

Be constructive, not dismissive.

Format:

CRITIQUE 1: [idea title]
WEAKNESS:
[real flaw]

UPGRADE:
[clear improvement]

CRITIQUE 2: [idea title]
WEAKNESS:
[real flaw]

UPGRADE:
[clear improvement]

CRITIQUE 3: [idea title]
WEAKNESS:
[real flaw]

UPGRADE:
[clear improvement]"""


CHIEF_PICK_SYSTEM = """You are a chief product officer.

Pick the idea with the highest real-world potential based on:
- usefulness
- feasibility
- differentiation

Avoid picking the safest idea by default.

Respond with:
PICK: [number] | REASON: [one sentence]"""


REFINER_SYSTEM = """You are a product refiner.

Take the selected idea and improve it into a clear, concrete concept.

Rules:
- Do not repeat the critique
- Be specific: who uses it, what happens step-by-step, and what outcome it produces
- Keep it realistic and buildable

Format:

REFINED IDEA: [title]
[4-5 sentences]"""


def concept_agent(domain: str, config: LLMConfig) -> str:
    return call_llm(config, CONCEPT_SYSTEM, f"Domain: {domain}")


def tension_agent(ideas: str, config: LLMConfig) -> str:
    return call_llm(config, TENSION_SYSTEM, f"Critique and upgrade these 3 ideas:\n\n{ideas}")


def chief_pick_agent(ideas: str, critiques: str, config: LLMConfig) -> str:
    prompt = f"IDEAS:\n{ideas}\n\nCRITIQUES:\n{critiques}"
    return call_llm(config, CHIEF_PICK_SYSTEM, prompt)


def refiner_agent(idea: str, critique: str, config: LLMConfig) -> str:
    prompt = f"IDEA:\n{idea}\n\nCRITIQUE:\n{critique}"
    return call_llm(config, REFINER_SYSTEM, prompt)
