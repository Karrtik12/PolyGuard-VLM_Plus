import os
import sys
import json
import httpx
from typing import Optional, Dict, Any
from PIL import Image


class VLMGatewayRouter:
    """
    Upstream VLM Gateway Proxy for PolyGuard-VLM_Plus.
    Routes safe requests to upstream VLMs (Ollama / LLaVA / OpenAI / vLLM)
    and streams/returns responses cleanly to the client.
    """
    def __init__(
        self,
        provider: str = "mock",
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: str = "llava-v1.6"
    ):
        self.provider = provider.lower()
        self.endpoint_url = endpoint_url or os.getenv("VLM_ENDPOINT_URL", "http://localhost:11434/api/generate")
        self.api_key = api_key or os.getenv("VLM_API_KEY", "")
        self.model_name = model_name

    async def route_safe_request(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        system_instruction: str = "You are a helpful multimodal assistant."
    ) -> Dict[str, Any]:
        """
        Routes safe request to upstream VLM backend and returns response payload.
        """
        if self.provider == "mock":
            return self._mock_vlm_response(prompt, image)
        elif self.provider == "ollama":
            return await self._call_ollama(prompt, image, system_instruction)
        elif self.provider in ["openai", "vllm"]:
            return await self._call_openai_compatible(prompt, image, system_instruction)
        else:
            return self._mock_vlm_response(prompt, image)

    def _mock_vlm_response(self, prompt: str, image: Optional[Image.Image]) -> Dict[str, Any]:
        """Built-in mock response generator for offline testing."""
        has_img = image is not None
        mock_text = (
            f"[PolyGuard-VLM_Plus Router -> Safe Request Approved]\n"
            f"VLM Output for prompt '{prompt}':\n"
            f"Analyzing input {'with uploaded visual context' if has_img else 'text prompt'}.\n"
            f"Response: Details provided safely without policy violations."
        )
        return {
            "status": "success",
            "vlm_response": mock_text,
            "provider": "mock-vlm-gateway",
            "model_used": self.model_name,
            "has_visual_input": has_img
        }

    async def _call_ollama(self, prompt: str, image: Optional[Image.Image], system_instruction: str) -> Dict[str, Any]:
        """Calls Ollama HTTP API endpoint."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_instruction,
            "stream": False
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self.endpoint_url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "status": "success",
                        "vlm_response": data.get("response", ""),
                        "provider": "ollama",
                        "model_used": self.model_name,
                        "raw": data
                    }
                else:
                    return {
                        "status": "error",
                        "vlm_response": f"Upstream VLM HTTP Error: {resp.status_code}",
                        "provider": "ollama"
                    }
        except Exception as e:
            return {
                "status": "fallback_mock",
                "vlm_response": f"Upstream VLM unreachable ({e}). Safe request verified.",
                "provider": "ollama_fallback"
            }

    async def _call_openai_compatible(self, prompt: str, image: Optional[Image.Image], system_instruction: str) -> Dict[str, Any]:
        """Calls OpenAI or vLLM compatible Chat Completions API."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self.endpoint_url, json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return {
                        "status": "success",
                        "vlm_response": content,
                        "provider": self.provider,
                        "model_used": self.model_name
                    }
                else:
                    return {
                        "status": "error",
                        "vlm_response": f"Upstream API Error: {resp.status_code}",
                        "provider": self.provider
                    }
        except Exception as e:
            return {
                "status": "fallback_mock",
                "vlm_response": f"Upstream VLM unreachable ({e}). Safe request verified.",
                "provider": f"{self.provider}_fallback"
            }


if __name__ == "__main__":
    import asyncio
    print("Testing VLMGatewayRouter...")
    router = VLMGatewayRouter(provider="mock")
    
    async def run_test():
        res = await router.route_safe_request(prompt="इस फोटो में क्या है?")
        print("Mock Router Response:")
        print(json.dumps(res, indent=2))
        assert res["status"] == "success"
        print("VLMGatewayRouter test PASSED!")
        
    asyncio.run(run_test())
