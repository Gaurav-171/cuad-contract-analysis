"""Bonus — semantic search over extracted clauses using embeddings.

Every extracted clause is embedded once (``build_index``) and stored in
``outputs/clause_embeddings.json``. ``search`` embeds a natural-language query
and ranks clauses by cosine similarity.
"""

import json

import numpy as np

from . import config, llm_client


def build_index(results: list[dict]) -> int:
    """Embed all extracted clauses and persist the index. Returns clause count."""
    entries = []
    for row in results:
        for clause_type in ("termination_clause", "confidentiality_clause", "liability_clause"):
            text = row.get(clause_type)
            if text:
                entries.append({
                    "contract_id": row["contract_id"],
                    "clause_type": clause_type,
                    "text": text,
                })
    if not entries:
        return 0

    vectors: list[list[float]] = []
    batch_size = 20
    for i in range(0, len(entries), batch_size):
        batch = [e["text"] for e in entries[i:i + batch_size]]
        vectors.extend(llm_client.embed_texts(batch))
        print(f"[search] Embedded {min(i + batch_size, len(entries))}/{len(entries)} clauses")

    for entry, vec in zip(entries, vectors):
        entry["embedding"] = vec

    config.EMBEDDINGS_JSON.write_text(json.dumps(entries))
    print(f"[search] Index with {len(entries)} clauses -> {config.EMBEDDINGS_JSON}")
    return len(entries)


def search(query: str, top_k: int = 5) -> list[dict]:
    """Return the top_k clauses most similar to the query."""
    entries = json.loads(config.EMBEDDINGS_JSON.read_text())
    matrix = np.array([e["embedding"] for e in entries])
    q = np.array(llm_client.embed_texts([query])[0])

    sims = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-9)
    order = np.argsort(-sims)[:top_k]
    return [
        {
            "score": round(float(sims[i]), 4),
            "contract_id": entries[i]["contract_id"],
            "clause_type": entries[i]["clause_type"],
            "text": entries[i]["text"][:500],
        }
        for i in order
    ]
