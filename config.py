"""
Central configuration for the Inference Value Theorem project.
"""

import os
import itertools
from pathlib import Path

# ---------------------------------------------------------------------------
# NIM endpoint
# ---------------------------------------------------------------------------
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ---------------------------------------------------------------------------
# API keys are read from environment variable NIM_API_KEYS as a comma-separated
# string. Do not commit provider keys to the repository.
# ---------------------------------------------------------------------------
_env_keys = os.environ.get("NIM_API_KEYS", "")
NIM_API_KEYS: list[str] = [k.strip() for k in _env_keys.split(",") if k.strip()]

# ---------------------------------------------------------------------------
# Round-robin key rotation
# ---------------------------------------------------------------------------
_key_cycle = itertools.cycle(NIM_API_KEYS)


def get_next_key() -> str:
    """Return the next API key in round-robin order."""
    if not NIM_API_KEYS:
        raise RuntimeError("Set NIM_API_KEYS before running collection scripts.")
    return next(_key_cycle)


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------
MODELS: dict[str, str] = {
    # --- Already collected / collecting ---
    "3B":              "meta/llama-3.2-3b-instruct",
    "8B":              "meta/llama-3.1-8b-instruct",
    "Super49B":        "nvidia/llama-3.3-nemotron-super-49b-v1",
    # "Ultra253B":       "nvidia/llama-3.1-nemotron-ultra-253b-v1",  # 404 — removed from NIM API
    "70B":             "meta/llama-3.3-70b-instruct",
    # --- New: Meta ---
    "Maverick17B":     "meta/llama-4-maverick-17b-128e-instruct",
    # --- New: Qwen ---
    "Qwen122B":        "qwen/qwen3.5-122b-a10b",
    "Qwen397B":        "qwen/qwen3.5-397b-a17b",
    "QwenNext80B":     "qwen/qwen3-next-80b-a3b-instruct",
    # --- New: Google Gemma ---
    "Gemma3n2B":       "google/gemma-3n-e2b-it",
    "Gemma3n4B":       "google/gemma-3n-e4b-it",
    # --- New: Mistral ---
    "MistralLarge675B":"mistralai/mistral-large-3-675b-instruct-2512",
    "MistralNemotron": "mistralai/mistral-nemotron",
    "MistralSmall119B":"mistralai/mistral-small-4-119b-2603",
    "Mixtral8x7B":     "mistralai/mixtral-8x7b-instruct-v0.1",
    "Mixtral8x22B":    "mistralai/mixtral-8x22b-instruct-v0.1",
    "Ministral14B":    "mistralai/ministral-14b-instruct-2512",
    # --- New: NVIDIA Nemotron ---
    "Super120B":       "nvidia/nemotron-3-super-120b-a12b",
    # --- New: Other families ---
    "Dracarys70B":     "abacusai/dracarys-llama-3.1-70b-instruct",
    "Stockmark100B":   "stockmark/stockmark-2-100b-instruct",
    # "SarvamM":         "sarvamai/sarvam-m",  # 400 Bad Request — disabled
    "GLM5":            "z-ai/glm5",
    "MiniMax":         "minimaxai/minimax-m2.5",
    "KimiK2":          "moonshotai/kimi-k2-instruct",
}

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
N_SAMPLES = 48
N_PROBLEMS = 500
RATE_LIMIT_PER_KEY = 35          # requests per minute per key
PILOT_K = 16
EVAL_N_VALUES = [1, 2, 4, 8, 16, 32, 48]

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR    = _PROJECT_ROOT / "data"
RAW_DIR     = DATA_DIR / "raw"
RESULTS_DIR = _PROJECT_ROOT / "results"

# Ensure directories exist at import time
for _d in (DATA_DIR, RAW_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
