"""Part B — concise contract summary (100-150 words).

The summary prompt receives the head of the contract (parties, recitals and
core obligations almost always appear early) plus the clauses already
extracted in Part A, so risk/penalty statements from anywhere in the document
inform the summary without re-sending the full text.
"""

from . import config, llm_client

_SCHEMA = {
    "type": "OBJECT",
    "properties": {"summary": {"type": "STRING"}},
    "required": ["summary"],
}

_PROMPT = """\
You are a legal contract analyst. Write a concise summary of the contract
below in 100-150 words. The summary must cover:
1. Purpose of the agreement (what kind of contract, between whom).
2. Key obligations of each party.
3. Notable risks or penalties (liability caps, termination triggers, indemnities).

Write in plain professional English, as one paragraph. Do not exceed 150 words.

Key clauses already extracted from this contract (may be null if absent):
- Termination: {termination}
- Confidentiality: {confidentiality}
- Liability: {liability}

Contract text (beginning of document):
---
{contract_text}
---
"""


def _clip(value: str | None, limit: int = 1200) -> str:
    return (value or "not found")[:limit]


def summarize(contract_text: str, clauses: dict) -> str:
    prompt = _PROMPT.format(
        termination=_clip(clauses.get("termination_clause")),
        confidentiality=_clip(clauses.get("confidentiality_clause")),
        liability=_clip(clauses.get("liability_clause")),
        contract_text=contract_text[: config.SUMMARY_CONTEXT_CHARS],
    )
    return llm_client.generate_json(prompt, _SCHEMA)["summary"].strip()
