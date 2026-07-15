"""Pipeline orchestration: load -> extract text -> LLM analysis -> outputs.

Results are checkpointed to ``outputs/results.json`` after every contract, so
an interrupted run (rate limits, network) resumes where it left off.
"""

import json
import time

import pandas as pd

from . import clause_extractor, config, data_loader, preprocess, summarizer


def _load_checkpoint() -> list[dict]:
    if config.RESULTS_JSON.exists():
        return json.loads(config.RESULTS_JSON.read_text())
    return []


def _save(results: list[dict]) -> None:
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    config.RESULTS_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    pd.DataFrame(results).to_csv(config.RESULTS_CSV, index=False)


def run(num_contracts: int | None = None, build_search_index: bool = True) -> list[dict]:
    num_contracts = num_contracts or config.NUM_CONTRACTS
    pdf_paths = data_loader.load_contracts(num_contracts)

    results = _load_checkpoint()
    done = {row["contract_id"] for row in results}
    processed = len(results)

    for pdf_path in pdf_paths:
        if processed >= num_contracts:
            break
        contract_id = pdf_path.stem
        if contract_id in done:
            continue

        raw = preprocess.extract_pdf_text(pdf_path)
        text = preprocess.normalize_text(raw)
        if not preprocess.is_usable(text):
            print(f"[skip] {contract_id}: too little extractable text "
                  f"({len(text)} chars, likely a scanned PDF)")
            continue

        print(f"[{processed + 1}/{num_contracts}] {contract_id} "
              f"({len(text):,} chars)")
        start = time.time()
        try:
            clauses = clause_extractor.extract_clauses(text)
            summary = summarizer.summarize(text, clauses)
        except Exception as exc:  # never let one contract kill the batch
            print(f"[error] {contract_id}: {type(exc).__name__}: {exc}")
            continue

        results.append({
            "contract_id": contract_id,
            "summary": summary,
            "termination_clause": clauses["termination_clause"],
            "confidentiality_clause": clauses["confidentiality_clause"],
            "liability_clause": clauses["liability_clause"],
            "num_chars": len(text),
        })
        processed += 1
        _save(results)
        print(f"        done in {time.time() - start:.1f}s")

    print(f"\n[pipeline] {len(results)} contracts -> {config.RESULTS_JSON} / {config.RESULTS_CSV}")

    if build_search_index:
        from . import semantic_search
        semantic_search.build_index(results)

    return results
