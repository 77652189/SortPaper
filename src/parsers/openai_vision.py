from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def call_openai_vision(
    image_b64: str,
    mime_type: str,
    prompt: str,
    *,
    model: str | None = None,
    timeout: float = 60.0,
    max_output_tokens: int = 2048,
) -> str:
    """Analyze one image with an OpenAI-compatible vision model."""
    return call_openai_vision_images(
        images=[{"image_b64": image_b64, "mime_type": mime_type}],
        prompt=prompt,
        model=model,
        timeout=timeout,
        max_output_tokens=max_output_tokens,
    )


def call_openai_vision_images(
    images: list[dict[str, str]],
    prompt: str,
    *,
    model: str | None = None,
    timeout: float = 60.0,
    max_output_tokens: int = 2048,
) -> str:
    """Analyze one or more images with an OpenAI-compatible vision model."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.error("OPENAI_API_KEY is not set")
        return ""
    if not images:
        return ""

    from openai import OpenAI

    client_kwargs = {"api_key": api_key, "timeout": timeout}
    base_url = os.getenv("OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE", "")).strip()
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    model_name = model or os.getenv("OPENAI_VISION_MODEL", "gpt-5.5")
    image_inputs = [
        {
            "data_url": f"data:{item.get('mime_type', 'image/png')};base64,{item.get('image_b64', '')}",
        }
        for item in images
        if item.get("image_b64")
    ]
    if not image_inputs:
        return ""
    wire_api = os.getenv("OPENAI_VISION_WIRE_API", "responses").strip().lower()

    if wire_api != "chat":
        try:
            from src.judge.llm_runtime import run_llm_call

            response = run_llm_call(
                lambda: client.responses.create(
                    model=model_name,
                    input=[{
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}] + [
                            {"type": "input_image", "image_url": item["data_url"], "detail": "high"}
                            for item in image_inputs
                        ],
                    }],
                    max_output_tokens=max_output_tokens,
                ),
                label="openai-vision-responses",
            )
            text = getattr(response, "output_text", "")
            if text:
                return text.strip()
            return _extract_responses_text(response).strip()
        except Exception as exc:
            logger.warning("OpenAI Responses vision call failed, falling back to chat: %s", exc)

    try:
        from src.judge.llm_runtime import run_llm_call

        response = run_llm_call(
            lambda: client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}] + [
                        {"type": "image_url", "image_url": {"url": item["data_url"], "detail": "high"}}
                        for item in image_inputs
                    ],
                }],
                max_tokens=max_output_tokens,
            ),
            label="openai-vision-chat",
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.error("OpenAI vision call failed: %s", exc)
        return ""


def _extract_responses_text(response: object) -> str:
    parts: list[str] = []
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            text = getattr(content, "text", "")
            if text:
                parts.append(text)
    return "\n".join(parts)
