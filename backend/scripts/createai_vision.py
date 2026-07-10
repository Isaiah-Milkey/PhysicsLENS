"""ASU CreateAI vision endpoint — minimal client.

Run directly:  python backend/scripts/createai_vision.py path/to/image.png

Credentials come from .env at the PhysicsLENS root (CREATEAI_TOKEN,
CREATEAI_BASE_URL) via python-dotenv — never hardcode the token.
"""
import base64
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def query_vision(query: str, image_path: str, *,
                  model_provider: str = "openai", model_name: str = "gpt4o",
                  system_prompt: str = None, response_format: dict = None,
                  token: str = None, base_url: str = None) -> dict:
    token = token or os.environ["CREATEAI_TOKEN"]
    base_url = base_url or os.environ["CREATEAI_BASE_URL"]

    image_bytes = Path(image_path).read_bytes()
    ext = Path(image_path).suffix.lstrip(".") or "png"
    encoded_image = f"data:image/{ext};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

    payload = {
        "endpoint": "vision",
        "request_source": "override_params",
        "query": query,
        "image_file": encoded_image,
        "model_provider": model_provider,
        "model_name": model_name,
    }
    if system_prompt is not None:
        payload["model_params"] = {"system_prompt": system_prompt}
    if response_format is not None:
        payload["response_format"] = response_format

    resp = requests.post(
        base_url + "/query",
        json=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python backend/scripts/createai_vision.py <image_path>")
        sys.exit(1)
    result = query_vision(
        "What is shown in this image?",
        sys.argv[1],
        system_prompt="You are a helpful assistant that describes images.",
        response_format={"type": "json"},
    )
    print(result)
