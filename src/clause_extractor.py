"""Part A — LLM-powered clause extraction (termination, confidentiality, liability).

Design notes
------------
* Few-shot: the prompt carries two worked examples (one positive, one showing
  the correct behaviour when a clause is absent) — this measurably reduces
  hallucinated clauses and paraphrasing.
* Verbatim policy: the model is told to quote the contract, not paraphrase,
  so extractions can be traced back to the source document.
* Large contracts are processed in overlapping chunks; per-chunk hits are
  merged with de-duplication.
"""

from . import llm_client, preprocess

CLAUSE_TYPES = ("termination_clause", "confidentiality_clause", "liability_clause")

_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "termination_clause": {"type": "STRING", "nullable": True},
        "confidentiality_clause": {"type": "STRING", "nullable": True},
        "liability_clause": {"type": "STRING", "nullable": True},
    },
    "required": list(CLAUSE_TYPES),
}

_FEW_SHOT = """\
EXAMPLE 1
Contract excerpt:
"8.1 Term. This Agreement shall commence on the Effective Date and continue for two (2) years.
8.2 Either party may terminate this Agreement upon thirty (30) days written notice if the other party materially breaches any provision and fails to cure such breach within the notice period.
9.1 Each party agrees to hold the other party's Confidential Information in strict confidence and not to disclose it to any third party without prior written consent."
Correct output:
{"termination_clause": "Either party may terminate this Agreement upon thirty (30) days written notice if the other party materially breaches any provision and fails to cure such breach within the notice period.", "confidentiality_clause": "Each party agrees to hold the other party's Confidential Information in strict confidence and not to disclose it to any third party without prior written consent.", "liability_clause": null}

EXAMPLE 2
Contract excerpt:
"5.2 IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES, AND EACH PARTY'S AGGREGATE LIABILITY SHALL NOT EXCEED THE FEES PAID IN THE PRECEDING TWELVE (12) MONTHS."
Correct output:
{"termination_clause": null, "confidentiality_clause": null, "liability_clause": "IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES, AND EACH PARTY'S AGGREGATE LIABILITY SHALL NOT EXCEED THE FEES PAID IN THE PRECEDING TWELVE (12) MONTHS."}
"""

_PROMPT = """\
You are a legal contract analyst. Extract the following clause types from the
contract text below.

Clause definitions:
- termination_clause: conditions under which the agreement may or will end
  (termination for breach/convenience, expiration, notice periods, effects of termination).
- confidentiality_clause: obligations to keep information confidential /
  non-disclosure obligations.
- liability_clause: limitation of liability, liability caps, exclusion of
  damages, indemnification for losses.

Rules:
1. Quote the clause text VERBATIM from the contract. Do not paraphrase or summarize.
2. If a clause type spans multiple relevant provisions, join the most important
   ones with " [...] " (keep the total under ~400 words per clause type).
3. If a clause type is genuinely absent from this text, return null for it.
4. Never invent text that is not in the contract.

{few_shot}
Now extract from this contract text:
---
{contract_text}
---
"""


def _extract_from_chunk(chunk: str) -> dict:
    prompt = _PROMPT.format(few_shot=_FEW_SHOT, contract_text=chunk)
    return llm_client.generate_json(prompt, _SCHEMA)


def _merge(results: list[dict]) -> dict:
    """Merge per-chunk extractions, de-duplicating overlapping hits."""
    merged: dict[str, str | None] = {k: None for k in CLAUSE_TYPES}
    for result in results:
        for key in CLAUSE_TYPES:
            value = (result.get(key) or "").strip() or None
            if value is None:
                continue
            if merged[key] is None:
                merged[key] = value
            elif value[:200] not in merged[key]:  # skip overlap duplicates
                merged[key] = f"{merged[key]} [...] {value}"
    return merged


def extract_clauses(contract_text: str) -> dict:
    """Extract the three clause types from a full (possibly long) contract."""
    chunks = preprocess.chunk_text(contract_text)
    results = [_extract_from_chunk(chunk) for chunk in chunks]
    return _merge(results)
