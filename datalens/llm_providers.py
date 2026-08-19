"""
LLM Provider Abstraction Layer for DataLens AI
Allows plug-and-play LLM integrations (Gemini, Ollama, OpenAI, etc.)
while keeping the core AI Engine agnostic to the specific API.
"""

from abc import ABC, abstractmethod
import json
import os
import urllib.request
import urllib.error
from typing import Optional, Tuple
from .logger import app_logger

class BaseLLMProvider(ABC):
    """Abstract base class for all LLM integrations."""
    
    @abstractmethod
    def generate(self, prompt: str) -> Optional[str]:
        """Generates text from a given prompt. Returns None on failure."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the provider is configured and available to use."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Returns the name of the LLM provider (e.g. 'Gemini', 'Local Ollama')."""
        pass


class GeminiProvider(BaseLLMProvider):
    """Google Gemini implementation."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        self.client = None
        self._init_client()
        
    def _init_client(self):
        if not self.api_key:
            return
            
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            app_logger.info("Initialized Google GenAI client.")
        except Exception:
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                self.client = legacy_genai.GenerativeModel("gemini-1.5-flash")
                app_logger.info("Initialized legacy Google GenerativeAI client.")
            except Exception as e:
                app_logger.warning(f"Could not initialize LLM client: {e}")
                self.client = None

    def is_available(self) -> bool:
        return self.client is not None

    def get_provider_name(self) -> str:
        return "Gemini"

    def generate(self, prompt: str) -> Optional[str]:
        if not self.is_available():
            return None

        # Candidate models to try in sequence
        models_to_try = [self.model_name, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]
        models_to_try = list(dict.fromkeys(models_to_try))

        for m in models_to_try:
            try:
                # Priority 1: models.generate_content
                if hasattr(self.client, "models") and hasattr(self.client.models, "generate_content"):
                    resp = self.client.models.generate_content(model=m, contents=prompt)
                    if resp and resp.text:
                        return resp.text.strip()

                # Priority 2: interactions.create
                if hasattr(self.client, "interactions") and hasattr(self.client.interactions, "create"):
                    inter = self.client.interactions.create(model=m, input=prompt)
                    if inter and hasattr(inter, "output_text") and inter.output_text:
                        return inter.output_text.strip()

                # Priority 3: Legacy generate_content
                if hasattr(self.client, "generate_content"):
                    resp = self.client.generate_content(prompt)
                    if resp and resp.text:
                        return resp.text.strip()
            except Exception as e:
                err_str = str(e)
                app_logger.warning(f"Model {m} attempt: {e}")
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    break
                continue

        return None


class LocalOllamaProvider(BaseLLMProvider):
    """Local Ollama-compatible HTTP endpoint implementation."""
    
    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
        self.base_url = (
            os.getenv("LOCAL_AI_BASE_URL")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        ).rstrip("/")
        self.model = os.getenv("LOCAL_MODEL_NAME") or os.getenv("OLLAMA_MODEL") or "llama3.1"
        self.enabled = os.getenv("ENABLE_LOCAL_AI", "1").strip().lower() not in {"0", "false", "no", "off"}
        
    def is_available(self) -> bool:
        return self.enabled

    def get_provider_name(self) -> str:
        return "Local Ollama"

    def generate(self, prompt: str) -> Optional[str]:
        if not self.is_available():
            return None
            
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 900,
            },
        }).encode("utf-8")

        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
                text = data.get("response") or data.get("text")
                if text and text.strip():
                    return text.strip()
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            app_logger.info(f"Local AI unavailable at {self.base_url} using model '{self.model}': {exc}")

        return None
