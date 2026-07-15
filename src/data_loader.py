"""Download the CUAD dataset and sample a subset of contract PDFs.

The official CUAD v1 archive (Zenodo) contains 510 contract PDFs organised by
contract category. We sample deterministically (fixed seed) across the whole
collection so runs are reproducible, and copy the selected PDFs into
``data/contracts/``.
"""

import random
import shutil
import zipfile
from pathlib import Path

import requests

from . import config


def download_cuad_zip(dest: Path) -> Path:
    """Download the CUAD v1 archive if not already present."""
    if dest.exists():
        print(f"[data] Using cached archive: {dest}")
        return dest
    print(f"[data] Downloading CUAD v1 (~100 MB) from {config.CUAD_ZIP_URL} ...")
    with requests.get(config.CUAD_ZIP_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print("[data] Download complete.")
    return dest


def sample_contracts(zip_path: Path, n: int, out_dir: Path) -> list[Path]:
    """Extract a deterministic sample of n contract PDFs from the archive.

    We over-sample (2x) so that scanned/image-only PDFs discovered later in
    the text-extraction step can be replaced without re-downloading.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in out_dir.iterdir() if p.suffix.lower() == ".pdf")
    if len(existing) >= n:
        print(f"[data] {len(existing)} PDFs already in {out_dir}, skipping extraction.")
        return existing

    with zipfile.ZipFile(zip_path) as zf:
        pdf_members = [
            m for m in zf.namelist()
            if m.lower().endswith(".pdf") and "full_contract_pdf" in m.lower()
        ]
        rng = random.Random(config.RANDOM_SEED)
        rng.shuffle(pdf_members)
        picked = pdf_members[: n * 2]  # over-sample for fallback
        for member in picked:
            target = out_dir / Path(member).name
            if target.exists():
                continue
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    pdfs = sorted(p for p in out_dir.iterdir() if p.suffix.lower() == ".pdf")
    print(f"[data] Extracted {len(pdfs)} candidate PDFs to {out_dir}")
    return pdfs


def load_contracts(n: int | None = None) -> list[Path]:
    """End-to-end: ensure the archive exists and return sampled PDF paths."""
    n = n or config.NUM_CONTRACTS
    zip_path = config.DATA_DIR / "CUAD_v1.zip"
    download_cuad_zip(zip_path)
    return sample_contracts(zip_path, n, config.CONTRACTS_DIR)
