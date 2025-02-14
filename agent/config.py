import os
import httpx
from dotenv import load_dotenv
load_dotenv(override=True)

USE_FALLBACK_API = os.getenv("USE_FALLBACK_API", "false").lower() in ('true', '1', 'yes')

class APIConfig:
    """
    A class to manage API configurations for both primary and fallback endpoints.
    """
    def __init__(self) -> None:
        """
        Initialize with either primary or fallback configuration based on flag.
        """
        if USE_FALLBACK_API:
            # Fallback configuration (OpenAI)
            self.auth_token = os.getenv("OPENAI_API_KEY")
            self.inference_endpoint = "https://api.openai.com/v1/"
        else:
            # Primary configuration (Custom Proxy API)
            self.auth_token = os.getenv("AIPROXY_TOKEN")
            self.inference_endpoint = "http://aiproxy.sanand.workers.dev/openai/v1/"

        self.chat_model = "gpt-4o-mini"
        self.embedding_model = "text-embedding-3-small"

    def _test_chat_endpoint(self) -> str:
        """
        Tests the chat endpoint for the active configuration.
        """
        try:
            response = httpx.post(
                f"{self.inference_endpoint}chat/completions",
                headers={
                    "Authorization": f"Bearer {self.auth_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.chat_model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What is your name?"},
                    ],
                },
            )
            response.raise_for_status()
            return "-> Chat Endpoint Test OK ✅"
        except Exception as e:
            print("❌ Chat endpoint exception:")
            return str(e)

    def _test_embedding_endpoint(self) -> str:
        """
        Tests the embedding endpoint for the active configuration.
        """
        try:
            response = httpx.post(
                f"{self.inference_endpoint}embeddings",
                headers={
                    "Authorization": f"Bearer {self.auth_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.embedding_model,
                    "input": ["1+1", "How are you?"],
                },
            )
            response.raise_for_status()
            return "-> Embeddings Endpoint Test OK ✅"
        except Exception as e:
            print("❌ Embedding endpoint exception:")
            return str(e)

if __name__ == "__main__":
    api_config = APIConfig()
    print(f"Testing {'Fallback (OpenAI)' if USE_FALLBACK_API else 'Primary (AIProxy)'} API:")
    print(api_config._test_chat_endpoint())
    print(api_config._test_embedding_endpoint())