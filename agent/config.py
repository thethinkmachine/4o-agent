import os
import httpx
from dotenv import load_dotenv

load_dotenv()

class APIConfig:
    """
    A class to manage the configuration and testing of API endpoints for chat and embeddings.
    """

    def __init__(self) -> None:
        """
        Initializes APIConfig with default settings for chat and embedding endpoints.
        """
        self.ai_proxy_token: str = os.getenv("AIPROXY_TOKEN")
        self.chat_endpoint: str = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        self.chat_model: str = "gpt-4o-mini"
        self.embedding_endpoint: str = "http://aiproxy.sanand.workers.dev/openai/v1/embeddings"
        self.embedding_model: str = "text-embedding-3-small"

    def test_chat_endpoint(self) -> str:
        """
        Tests the chat endpoint by sending a request and checking the response.

        Returns:
            str: The result of the chat endpoint test.
        """
        try:
            response = httpx.post(
                self.chat_endpoint,
                headers={
                    "Authorization": f"Bearer {self.ai_proxy_token}",
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
            print("❌ An exception occurred:")
            return str(e)

    def test_embedding_endpoint(self) -> str:
        """
        Tests the embedding endpoint by sending a request and checking the response.

        Returns:
            str: The result of the embedding endpoint test.
        """
        try:
            response = httpx.post(
                self.embedding_endpoint,
                headers={
                    "Authorization": f"Bearer {self.ai_proxy_token}",
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
            print("❌ An exception occurred:")
            return str(e)

if __name__ == "__main__":
    config = APIConfig()
    print("Testing API Endpoints")
    print(config.test_chat_endpoint())
    print(config.test_embedding_endpoint())