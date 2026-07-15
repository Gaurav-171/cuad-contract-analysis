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
# One or more API keys (comma-separated). The free tier allows ~20 requests
# per day per project *per model*, so the client rotates through the
# GENERATION_MODELS fallback chain (and then the next key) as buckets run out.
GEMINI_API_KEYS = [k.strip() for k in os.getenv(
    "GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", "")).split(",") if k.strip()]
GENERATION_MODELS = [m.strip() for m in os.getenv(
    "GENERATION_MODELS",
    "gemini-3.5-flash,gemini-3-flash-preview,gemini-3.1-flash-lite,"
    "gemini-3.1-pro-preview,gemini-3-pro-preview,gemini-2.5-flash",
).split(",") if m.strip()]
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
