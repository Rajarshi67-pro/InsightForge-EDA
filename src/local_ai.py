"""Optional local LLM client helpers.

The app must remain useful without network credentials, so local model calls are
best-effort only and always fall back to deterministic analysis.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Optional

from .logger import app_logger


def local_ai_enabled() -> bool:
    """Returns whether local LLM calls are enabled by configuration."""
    flag = os.getenv("ENABLE_LOCAL_AI", "1").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def get_local_ai_config() -> tuple[str, str]:
    """Returns configured local model endpoint and model name."""
    base_url = (
        os.getenv("LOCAL_AI_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL")
        or "http://localhost:11434"
    ).rstrip("/")
    model = os.getenv("LOCAL_MODEL_NAME") or os.getenv("OLLAMA_MODEL") or "llama3.1"
    return base_url, model


def call_local_llm(prompt: str, timeout: float = 20.0) -> Optional[str]:
    """Calls an Ollama-compatible local model and returns generated text."""
    if not local_ai_enabled():
        return None

    base_url, model = get_local_ai_config()
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 900,
        },
    }).encode("utf-8")

    request = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            text = data.get("response") or data.get("text")
            if text and text.strip():
                return text.strip()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        app_logger.info(f"Local AI unavailable at {base_url} using model '{model}': {exc}")

    return None
