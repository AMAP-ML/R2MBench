"""
FocusCIR: MLLM API wrapper supporting only the openai backend.
Provides a unified callable interface for ReAct agent.

Uses async_batch_client for efficient API calls and dotenv for credential management.
"""

import base64
import io
import os
import time
import random
import requests
import concurrent.futures
from typing import List, Dict, Any, Optional
from PIL import Image

from dotenv import load_dotenv

# Lazy imports: only needed for non-gateway (OpenAI SDK) mode
OpenAI = None
batch_request_with_messages_sync = None

def _ensure_openai_imports():
    """Import openai and async_batch_client on demand."""
    global OpenAI, batch_request_with_messages_sync
    if OpenAI is None:
        from openai import OpenAI as _OpenAI
        OpenAI = _OpenAI
    if batch_request_with_messages_sync is None:
        from async_batch_client import batch_request_with_messages_sync as _batch
        batch_request_with_messages_sync = _batch

load_dotenv(os.environ.get("DOTENV_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")))

# Engines that use the hosted gateway (GPT, Claude, etc.)
GATEWAY_ENGINES = [
    "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-5.5-0424-global",
    "claude-sonnet-4-20250514", "claude-3.5-sonnet", "claude-opus-4-7",
    "o4-mini",
    "gemini-3.1-pro-preview",
]
# Credentials and endpoints are read from the environment (.env). Never hardcode.
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "")
GATEWAY_BASE_URL = os.getenv("GATEWAY_BASE_URL", "")

# Gemini may use its own key; GPT/Claude use the gateway key.
GEMINI_ENGINES = ["gemini-3.1-pro-preview"]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def encode_image_to_base64(image: Image.Image, max_size: Optional[int] = None) -> str:
    """Encode a PIL Image to a base64 data URL.

    By default (max_size=None) the image is sent losslessly: no downscaling
    and PNG encoding, so the model receives the original pixels. This matters
    for side-by-side comparison images where downscaling destroys fine detail
    (small/repeated objects, distant structures). Pass an integer max_size to
    cap the longest side if payload size becomes a concern.
    """
    if max_size is not None:
        width, height = image.size
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            image = image.resize(
                (int(width * scale), int(height * scale)), Image.LANCZOS
            )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


def encode_image_path_to_base64(image_path: str) -> str:
    """Encode an image file path to base64 data URL."""
    image = Image.open(image_path).convert("RGB")
    return encode_image_to_base64(image)


def _create_openai_clients(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    num_clients: int = 2,
) -> list:
    """
    Create a pool of OpenAI clients
    """
    _ensure_openai_imports()
    api_key = api_key or os.getenv("API_KEY", "")
    base_url = base_url or os.getenv("API_URL", "")

    if not api_key:
        raise ValueError(
            "No API key found. Set API_KEY in .env "
            "or pass api_key explicitly."
        )
    elif not base_url:
        raise ValueError(
            "No base URL found. Set API_URL in .env "
            "or pass base_url explicitly."
        )

    kwargs: Dict[str, Any] = {"api_key": api_key, "base_url": base_url}

    return [OpenAI(**kwargs) for _ in range(num_clients)]


def create_mllm_caller(
    engine: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    num_clients: int = 2,
    enable_thinking: Optional[bool] = None,
):
    """
    Factory function: create an MLLM caller.

    Args:
        engine: model name (e.g. "gpt-4o", "qwen3-vl-8b-instruct").
        temperature: sampling temperature.
        max_tokens: max output tokens.
        api_key: explicit API key (overrides .env).
        base_url: explicit base URL (overrides .env).
        num_clients: number of parallel async clients for batch mode.
        enable_thinking: whether to enable thinking mode (for Qwen3 etc.)

    Returns:
        callable(messages, images) -> str
    """
    return OpenAICaller(
        engine=engine,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        num_clients=num_clients,
        enable_thinking=enable_thinking,
    )


class OpenAICaller:
    """
    OpenAI-compatible API caller with multi-image ReAct support.

    Backed by async_batch_client for efficient parallel requests.
    Works with any OpenAI-compatible endpoint (OpenAI, DashScope,
    vLLM, Ollama, etc.) — just set the right base_url + api_key.
    """

    def __init__(
        self,
        engine: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        num_clients: int = 1,
        enable_thinking: Optional[bool] = None,
    ):
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.num_clients = num_clients

        # Determine if this engine uses the hosted gateway.
        # If env vars API_KEY/API_URL are set, use them as the gateway.
        env_api_key = os.getenv("API_KEY", "")
        env_api_url = os.getenv("API_URL", "")

        self.use_gateway = (
            (api_key is None and base_url is None and engine in GATEWAY_ENGINES)
            or (env_api_key and env_api_url)
        )

        if self.use_gateway:
            if env_api_key and env_api_url:
                # Use the env-specified gateway endpoint.
                self._gateway_api_key = env_api_key
                self._gateway_base_url = env_api_url.rstrip('/') + '/chat/completions'
            elif engine in GEMINI_ENGINES:
                self._gateway_api_key = GEMINI_API_KEY
                self._gateway_base_url = GATEWAY_BASE_URL
            else:
                self._gateway_api_key = GATEWAY_API_KEY
                self._gateway_base_url = GATEWAY_BASE_URL
            self.clients = None  # not needed for gateway mode
        else:
            self._gateway_api_key = None
            self._gateway_base_url = None
            self.clients = _create_openai_clients(
                api_key=api_key, base_url=base_url, num_clients=num_clients
            )

    # ---- Gateway (HTTP) helpers ----

    def _gateway_request(self, api_messages: List[Dict],
                         max_retries: int = 3, timeout: int = 120) -> str:
        """Send a single request to the hosted gateway via requests.post."""
        assert self._gateway_base_url, (
            "Gateway base URL is not set. Configure API_URL/API_KEY "
            "(or GATEWAY_BASE_URL) in your .env."
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._gateway_api_key}",
        }
        payload = {
            "model": self.engine,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
        }
        # Some models/gateways reject temperature=0; skip it for Claude, or set
        # GATEWAY_NO_TEMPERATURE=1 to omit it for a gateway that does not accept it.
        skip_temperature = os.getenv("GATEWAY_NO_TEMPERATURE", "") == "1"
        if "claude" not in self.engine and not skip_temperature:
            payload["temperature"] = self.temperature

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url=self._gateway_base_url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                )
                if response.status_code != 200:
                    print(f"[Gateway] HTTP {response.status_code}: "
                          f"{response.text[:500]}")
                response.raise_for_status()
                result = response.json()
                if "choices" not in result:
                    raise ValueError(
                        f"API response missing 'choices': "
                        f"{str(result)[:500]}"
                    )
                message = result["choices"][0]["message"]
                content = message.get("content") or message.get("reasoning_content") or ""
                return content
            except Exception as request_error:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt + random.random()
                    print(f"[Gateway] Attempt {attempt + 1} failed: "
                          f"{request_error}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[Gateway] All {max_retries} attempts failed: "
                          f"{request_error}")
                    return ""
        return ""

    # ---- Public API ----

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        images: List[Image.Image],
    ) -> str:
        """
        Call the MLLM with multi-turn messages and multiple images.

        Args:
            messages: conversation history [{role, content}, ...]
            images: list of PIL Images to include in the conversation.
                    images[0] = reference image (always in round 1).
                    images[1:] = focused views from tool calls.

        Returns:
            MLLM response text.
        """
        api_messages = self._build_messages(messages, images)

        if self.use_gateway:
            return self._gateway_request(api_messages)

        extra_body = None
        if self.enable_thinking is not None:
            extra_body = {
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking
                }
            }

        _ensure_openai_imports()
        responses = batch_request_with_messages_sync(
            clients=self.clients,
            messages_list=[api_messages],
            model=self.engine,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body=extra_body,
        )
        return responses[0] if responses else ""

    def batch_call(
        self,
        messages_list: List[List[Dict[str, Any]]],
        images_list: List[List[Image.Image]],
    ) -> List[str]:
        """
        Batch call the MLLM for multiple queries in parallel.
        Useful for processing an entire dataset efficiently.

        Args:
            messages_list: list of conversation histories.
            images_list: list of image lists, one per query.

        Returns:
            List of MLLM response strings.
        """
        all_api_messages = [
            self._build_messages(msgs, imgs)
            for msgs, imgs in zip(messages_list, images_list)
        ]

        if self.use_gateway:
            # breakpoint()
            # Parallel HTTP requests via thread pool
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.num_clients, len(all_api_messages))
            ) as executor:
                results = list(executor.map(
                    self._gateway_request, all_api_messages
                ))
            return results

        extra_body = None
        if self.enable_thinking is not None:
            extra_body = {
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking
                }
            }

        _ensure_openai_imports()
        return batch_request_with_messages_sync(
            clients=self.clients,
            messages_list=all_api_messages,
            model=self.engine,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body=extra_body,
        )

    def _build_messages(
        self,
        messages: List[Dict[str, Any]],
        images: List[Image.Image],
    ) -> List[Dict]:
        """Build OpenAI-compatible messages with inline base64 images."""
        api_messages = []
        image_idx = 0

        # Index of the last user message, which receives any leftover images.
        user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
        last_user_index = user_indices[-1] if user_indices else -1

        for msg_index, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                api_messages.append({"role": "system", "content": content})

            elif role == "user":
                content_parts = [{"type": "text", "text": content}]

                # Attach one image to this user message; the last user message
                # absorbs all remaining images (supports multi-image prompts).
                num_to_attach = 1
                if msg_index == last_user_index:
                    num_to_attach = len(images) - image_idx

                for _ in range(max(0, num_to_attach)):
                    if image_idx >= len(images):
                        break
                    image_data_url = encode_image_to_base64(images[image_idx])
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    })
                    image_idx += 1

                api_messages.append({"role": "user", "content": content_parts})

            elif role == "assistant":
                api_messages.append({"role": "assistant", "content": content})

        return api_messages
