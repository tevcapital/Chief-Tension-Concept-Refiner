import sys
import requests


# ─── CONFIG ───────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"
MODEL = "gpt-oss:20b"
TEMPERATURE = 0.3

# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a scrappy solo operator receiving a final decision from leadership.

Your job is ONLY to translate this decision into an immediate execution plan.

RULES:
- Accept the decision as correct. Do NOT question it, reinterpret it, or generate alternatives.
- You are starting from zero: no team, low budget, no infrastructure.
- Be concrete and immediate. Every sentence must describe a real action.
- No abstract language. No strategy discussion. No long-term roadmap.
- No "build a platform". No consulting jargon. No repetition of the idea in fancy terms.
- If a sentence sounds smart but unclear, rewrite it until it is obvious."""

EXECUTION_TEMPLATE = """You have received the following decision:

DECISION:
{primary_decision}

Produce an execution plan in EXACTLY this format. Do not add, remove, or rename sections.

EXECUTION PLAN:

WHAT THIS IS:
[1-2 sentences, plain language, no jargon]

WHO IT IS FOR:
[specific user or customer, be precise]

HOW YOU MAKE MONEY:
[exact monetization — price, model, frequency]

---

DAY 1-3:
[exact actions the operator takes on day 1, 2, and 3]

---

WEEK 1:
[what must be completed by end of week 1]

---

FIRST CUSTOMER:
[exactly how to find and close the first paying customer]

---

TOOLS:
[only the essential tools needed — list them]

---

RISKS:
[2-4 real risks that could kill this]

---

KILL CRITERIA:
[specific conditions under which you stop and move on]"""


def execute(primary_decision: str) -> str:
    prompt = EXECUTION_TEMPLATE.format(primary_decision=primary_decision.strip())

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": TEMPERATURE,
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python3 execution_agent.py "your primary decision here"')
        sys.exit(1)
    decision = " ".join(sys.argv[1:])
    result = execute(decision)
    print(result)
