"""Central configuration for the CUAD contract-analysis pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONTRACTS_DIR = DATA_DIR / "contracts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RESULTS_JSON = OUTPUTS_DIR / "results.json"
RESULTS_CSV = OUTPUTS_DIR / "results.csv"
EMBEDDINGS_JSON = OUTPUTS_DIR / "clause_embeddings.json"

# --- Dataset -------------------------------------------------------------
CUAD_ZIP_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"
NUM_CONTRACTS = int(os.getenv("NUM_CONTRACTS", "50"))
RANDOM_SEED = 42  # deterministic contract sampling
MIN_TEXT_CHARS = 1500  # skip scanned/empty PDFs that yield less text than this

# --- LLM (Google Gemini) --------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-3.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Free-tier friendly pacing / retries
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "9"))
MAX_RETRIES = 5

# --- Large-text handling ---------------------------------------------------
# Contracts longer than CHUNK_CHARS are split into overlapping chunks and the
# per-chunk extractions are merged. Keeps every request comfortably inside
# free-tier token-per-minute limits.
CHUNK_CHARS = 100_000
CHUNK_OVERLAP = 5_000
SUMMARY_CONTEXT_CHARS = 60_000
