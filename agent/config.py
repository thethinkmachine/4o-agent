import os
import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

"""
Environment variables:
- AIPROXY_TOKEN: API Proxy token.
- USE_CUSTOM_API: If true then use custom API configuration.
- CUSTOM_BASE_URL: Custom API base URL.
- CUSTOM_API_KEY: Custom API key.
- CHAT_MODEL: Chat model name.
- EMBEDDING_MODEL: Embedding model name.
"""

# API configuration constants
USE_CUSTOM_API = os.getenv("USE_CUSTOM_API", "false").lower() in ('true', '1', 'yes') # If true then use below given configuration
CUSTOM_BASE_URL = os.getenv("CUSTOM_BASE_URL", "https://api.openai.com/v1/")
CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY")

# Backend model configuration
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


class APIConfig:
    """
    Manages API configurations for both primary and custom endpoints.
    """
    def __init__(self) -> None:
        if not USE_CUSTOM_API:
            self.inference_endpoint = "https://aiproxy.sanand.workers.dev/openai/v1/" # Primary API endpoint
            self.auth_token = os.getenv("AIPROXY_TOKEN") # Primary API auth token
        else:
            self.inference_endpoint = CUSTOM_BASE_URL
            self.auth_token = CUSTOM_API_KEY
            os.environ["CUSTOM_FLAG"] = "1"

        self.chat_model = CHAT_MODEL
        self.embedding_model = EMBEDDING_MODEL

    def _test_chat_endpoint(self) -> str:
        """
        Tests the chat endpoint.
        """
        url = f"{self.inference_endpoint.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is your name?"},
            ],
        }
        try:
            print(f"POST {url}")
            response = httpx.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return "🟢 Chat Endpoint Test: OK ✅"
        except httpx.HTTPStatusError as http_err:
            print("🔴 Chat Endpoint Test: FAILED ❌")
            return str(http_err)
        except httpx.RequestError as req_err:
            print("🟡 Chat Endpoint Test: Timeout ⌛")
            return str(req_err)

    def _test_embedding_endpoint(self) -> str:
        """
        Tests the embedding endpoint.
        """
        url = f"{self.inference_endpoint.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.embedding_model,
            "input": "What is your name?",
        }
        try:
            print(f"POST {url}")
            response = httpx.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return "🟢 Embedding Endpoint Test: OK ✅"
        except httpx.HTTPStatusError as http_err:
            print("🔴 Embedding Endpoint Test: FAILED ❌")
            return str(http_err)
        except httpx.RequestError as req_err:
            print("🟡 Embedding Endpoint Test: Timeout ⌛")
            return str(req_err)

if __name__ == "__main__":
    api_config = APIConfig()
    print(f"🧪 Testing {'Custom' if USE_CUSTOM_API else 'Primary'} API:")
    print(api_config._test_chat_endpoint())
    print(api_config._test_embedding_endpoint())