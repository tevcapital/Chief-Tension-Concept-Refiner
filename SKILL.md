---
name: chief-tension-concept-refiner
description: Generates bold, OpenClaw-native use case ideas for a given domain using a 4-agent pipeline (Concept → Tension → Chief → Refiner). Run with /chief-tension-concept-refiner [domain].
version: 1.0.0
user-invocable: true
metadata.openclaw: {"requires": {"bins": ["python3"]}, "emoji": "🧠", "skillKey": "chief-tension-concept-refiner"}
---

# Chief-Tension-Concept-Refiner

A 4-agent ideation system that generates, critiques, selects, and refines bold use case ideas for OpenClaw in a given domain.

## Agents

- **Concept Agent** — generates 3 use cases that require OpenClaw's always-on, local, multi-app nature
- **Tension Agent** — critiques each idea and proposes a concrete upgrade
- **Chief Agent** — picks the boldest, most compelling idea (thinks like Elon Musk)
- **Refiner Agent** — merges the original idea with the upgrade into a final, polished use case

## Usage

When the user runs `/chief-tension-concept-refiner [domain]`, execute:

```bash
cd {baseDir} && python3 main.py "[domain]"
```

Pass the user's domain argument directly as the first argument to `main.py`.

## Requirements

- Python 3 must be installed
- Ollama must be running locally at `http://localhost:11434`
- Model `qwen3.5:4b` must be pulled: `ollama pull qwen3.5:4b`
- Install Python dependencies: `pip install requests`

## Example

```
/chief-tension-concept-refiner healthcare
/chief-tension-concept-refiner "solo entrepreneurs"
/chief-tension-concept-refiner legal
```
