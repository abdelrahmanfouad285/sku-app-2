"""Vision extraction via a vision LLM.

Supports two backends:
  - Anthropic Claude (cloud, paid)
  - Ollama local models (free, requires Ollama running locally)

The retry decorator handles transient errors and malformed JSON.
"""

import json
import os
import re
from typing import Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    retry_if_exception_type,
)

# ---------------------------------------------------------------------------
# Vision capability registry — single source of truth for "which Ollama models
# can handle image/multimodal input". Used BEFORE any HTTP call to avoid
# the wasted 400 -> fallback -> 400 cycle.
# ---------------------------------------------------------------------------

VISION_KEYWORDS = ("llava", "moondream", "vision", "bakllava", "minicpm", "qwen", "mllama")
# Best default for this SKU agent: auto-switch to a vision model so the
# pipeline still extracts quantity/size instead of failing. Make it
# configurable without code change.
VISION_FALLBACK_MODEL = os.getenv("OLLAMA_VISION_FALLBACK", "llava:7b")
VISION_MODE = os.getenv("OLLAMA_VISION_MODE", "auto_switch")  # auto_switch | fail_fast | strip


def is_vision_model(model: str) -> bool:
    """Return True if the model name looks like a vision-capable Ollama model."""
    if not model:
        return False
    m = model.lower()
    return any(kw in m for kw in VISION_KEYWORDS)


def _resolve_vision_fallback(requested: str) -> str | None:
    """Return a fallback vision model to use, or None if none configured/valid."""
    if VISION_MODE != "auto_switch":
        return None
    fb = (VISION_FALLBACK_MODEL or "").strip()
    if not fb or not is_vision_model(fb):
        return None
    # Avoid infinite loop if fallback is same as requested
    if fb.lower() == requested.lower():
        return None
    return fb


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


SYSTEM_PROMPT = (
    "You are an expert retail product-data extraction agent for Middle-Eastern "
    "and Egyptian packaging, which often mixes Arabic and English. Given a product "
    "photo, identify every packaged retail product visible — cans, bottles, boxes, "
    "bags, etc. Include every distinct product you can see, even if some text is "
    "blurry or partially hidden — use null for unreadable fields, but always count "
    "the visible package. Only omit if you are sure the object is not a retail "
    "product. If a field is unreadable or absent, use null. NEVER invent, guess, or "
    "hallucinate values — prefer null over guessing. Preserve Arabic characters "
    "exactly; never transliterate or translate. Respond with JSON only."
)

USER_PROMPT_TEMPLATE = """You are a retail inventory data-entry assistant. You will be shown a photo that may contain ONE product or MANY products (e.g. a shelf, fridge, or multiple items in frame).

Follow these steps internally before answering:
1. Scan the entire image systematically, left to right, top to bottom, row by row.
2. Identify every visually DISTINCT product-flavor-size combination. Two cans of the same brand but different flavor or size (e.g. "Gorilla Mango Coconut" vs "Gorilla Watermelon Melon") are DIFFERENT items and must be listed separately.
3. For each distinct item, count how many visible units you can see of exactly that item. If units are stacked, partially hidden, or cut off at the frame edge, still count what is visibly identifiable, and note the uncertainty in "notes".
4. Do NOT merge similar-looking items into one entry. Do NOT skip items because they look similar to one already listed. If you are unsure whether two cans are the same or different, list them as separate items and set confidence to "low".

STRICT RULES — read carefully:
- Only report a brand, product name, flavor, or SKU if you can literally read it printed on the packaging in the image. NEVER infer, guess, or generalize a brand from the product category (e.g. do not write "Energizer" or "generic energy drink" just because a can looks like an energy drink — if you cannot read the brand text clearly, set the field to null and confidence to "low").
- Text may be in Arabic, English, or both. Extract whichever is legible; if only Arabic is visible, put the Arabic text in "product_name" as-is — do not translate or invent an English equivalent.
- The "sku" field must NEVER be null. If an official SKU is not printed on the package, you MUST generate a short logical code based on the brand, product name, and flavor (e.g., "GOR-MANG-330" or "PEPSI-MAX").
- The "quantity_or_size" field must NEVER be null. Extract net weight or volume if printed (e.g., "330ml", "500g", "1L"). If not visibly printed, you MUST estimate it based on the container type (e.g., "approx 330ml can" or "large bottle").
- If the image is too blurry, too small, or too obstructed to read a field, set it to null. A null field is correct behavior, not a failure.
- Confidence must reflect true certainty:
  - "high": brand, product name, and flavor/size are all clearly legible
  - "medium": most fields legible, one or two uncertain or partially obscured
  - "low": guessing, heavy obstruction, ambiguous count, or duplicate-vs-distinct uncertainty
- If more than 4 distinct items are detected in one photo, set confidence to "low" on ALL items in that photo, regardless of individual legibility — dense multi-item photos should always be flagged for human review.

OUTPUT FORMAT — return ONLY valid JSON, no markdown fences, no commentary, no explanation text before or after:
{{
  "item_count": <integer, number of distinct items listed below>,
  "items": [
    {{
      "sku": string,
      "brand": string or null,
      "product_name": string or null,
      "flavor_or_variant": string or null,
      "quantity_or_size": string,
      "visible_unit_count": integer or null,
      "confidence": "high" | "medium" | "low",
      "notes": string
    }}
  ]
}}

If you cannot confidently identify ANY products in the image, return {{"item_count": 0, "items": []}} rather than guessing.
"""


def _parse_json(text: str) -> dict[str, Any]:
    """Strip markdown fences if present, locate the first {...} block, parse."""
    if not text:
        print(f"[llm] JSON parse failed: empty response from LLM (raw text is empty)", flush=True)
        raise ValueError("Empty response from LLM")

    m = _FENCE_RE.search(text)
    candidate = m.group(1) if m else text

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        print(f"[llm] JSON parse failed: no JSON object found. Raw text (first 500 chars): {text[:500]!r}", flush=True)
        raise ValueError(f"No JSON object in response: {text[:200]!r}")
    candidate = candidate[start:end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        print(f"[llm] JSON parse failed: {e}. Raw candidate (first 500 chars): {candidate[:500]!r}", flush=True)
        print(f"[llm] Full raw text (first 1000 chars): {text[:1000]!r}", flush=True)
        raise ValueError(f"Invalid JSON from LLM: {e}. Raw: {candidate[:200]!r}") from e


# ---------------------------------------------------------------------------
# Anthropic Claude backend
# ---------------------------------------------------------------------------

CLAUDE_MODEL_NAME = "claude-3-5-sonnet-20241022"


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type(Exception),
)
def extract_with_claude(
    client, image_b64: str
) -> dict[str, Any]:
    """Send the image to Anthropic Claude and return the parsed JSON."""
    import anthropic  # imported lazily so the local backend doesn't need this package

    user_prompt = USER_PROMPT_TEMPLATE

    message = client.messages.create(
        model=CLAUDE_MODEL_NAME,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
    )

    parts = []
    for block in message.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    text = "".join(parts).strip()

    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        raise ValueError("Top-level JSON is not an object")
    parsed.setdefault("item_count", 0)
    parsed.setdefault("items", [])
    return parsed


# ---------------------------------------------------------------------------
# Ollama local backend
# ---------------------------------------------------------------------------

class ModelNotVisionError(RuntimeError):
    """Raised when both Ollama endpoints reject an image — model is text-only. Not retried."""


def _is_ollama_400(exc: Exception) -> bool:
    """True when the OpenAI-compatible request failed with HTTP 400 (Bad Request).

    Catches both the SDK's status_code attribute and the wrapped error text so
    the native fallback triggers reliably for any 400, not just multimodal ones.
    """
    status = getattr(exc, "status_code", None)
    if status == 400:
        return True
    msg = str(exc).lower()
    return "400" in msg or "bad request" in msg


def _is_not_vision_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "multimodal" in msg or "does not support" in msg


def _is_retryable_error(exc: Exception) -> bool:
    """Only transient errors should be retried. 4xx client errors are NOT retryable."""
    # Explicit non-retryable marker
    if isinstance(exc, ModelNotVisionError):
        return False
    status = getattr(exc, "status_code", None)
    # 5xx and connection/timeout are retryable
    if isinstance(status, int):
        if 500 <= status < 600:
            return True
        if 400 <= status < 500:
            # Special case: 500 with mllama architecture on old Ollama is NOT retryable
            # (needs `winget upgrade Ollama.Ollama`)
            return False
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "mllama" in msg or "unknown model architecture" in msg:
        return False
    if "connection" in name or "timeout" in name:
        return True
    if "connection" in msg or "timed out" in msg or "timeout" in msg:
        return True
    return False


def _is_mllama_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "mllama" in msg or "unknown model architecture" in msg


def validate_models_at_startup(client, configured_models: list[str] | None = None) -> None:
    """Startup/config validation: confirm configured model(s) exist and log capabilities.

    Call this once when the app loads (e.g. in Streamlit sidebar init) so
    mismatches are caught early. Never raises — just logs.
    """
    try:
        models = client.models.list()
        models_data = getattr(models, "data", []) or []
        installed = sorted({getattr(m, "id", "?") for m in models_data if getattr(m, "id", None)})
        if not installed:
            print("[llm] Startup check: no models installed. Pull a vision model: ollama pull llava:7b", flush=True)
            return
        # Log all installed with capability tag
        tagged = [f"{m} ({'vision' if is_vision_model(m) else 'text-only'})" for m in installed]
        print(f"[llm] Startup check: {len(installed)} model(s) installed: {', '.join(tagged)}", flush=True)
        if configured_models:
            for cfg in configured_models:
                # Handle :latest tag fuzzy match
                found = any(cfg == im or im.startswith(cfg + ":") or cfg.startswith(im + ":") for im in installed)
                if not found:
                    print(f"[llm] Startup check WARNING: configured model '{cfg}' not found in installed {installed}. Pull it: ollama pull {cfg}", flush=True)
                else:
                    # Find actual installed name for capability check
                    actual = next((im for im in installed if cfg == im or im.startswith(cfg + ":") or cfg.startswith(im + ":")), cfg)
                    cap = "vision" if is_vision_model(actual) else "text-only"
                    print(f"[llm] Startup check: configured model '{cfg}' -> installed as '{actual}' ({cap})", flush=True)
                    if not is_vision_model(actual):
                        fb = _resolve_vision_fallback(actual) or "none"
                        print(f"[llm] Startup check WARNING: '{actual}' is text-only and will be rejected for image input. Configured vision fallback: {fb}", flush=True)
    except Exception as exc:
        print(f"[llm] Startup check: could not list models at {getattr(client, 'base_url', '?')}: {exc}", flush=True)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_retryable_error),
)
def extract_with_ollama(
    client, model: str, image_b64: str
) -> dict[str, Any]:
    """Send the image to a local Ollama model and return the parsed JSON.

    Tries the OpenAI-compatible /v1/chat/completions endpoint first (works for
    llama3.2-vision, moondream). Falls back to native only for genuine
    endpoint incompatibilities (e.g. llava on OpenAI shim), NOT for
    'model can't do this at all' errors.
    """
    # --- 1. Capability check BEFORE any HTTP call ---
    # This request always contains an image (image_b64 is required for SKU extraction).
    # If the selected model is text-only, don't waste two HTTP round-trips.
    if image_b64 and not is_vision_model(model):
        fb = _resolve_vision_fallback(model)
        # Single clear log line (requirement 5) instead of raw JSON stack trace
        print(
            f"[llm] Request rejected: model '{model}' does not support image input. "
            f"Configured vision fallback: {fb or 'none'} (mode={VISION_MODE}).",
            flush=True,
        )
        if fb:
            print(f"[llm] Auto-switching to vision fallback '{fb}' for this request.", flush=True)
            model = fb  # proceed with fallback model
        elif VISION_MODE == "strip":
            print(f"[llm] Stripping image content and sending text-only to '{model}' (not useful for SKU, will likely return empty).", flush=True)
            # Fall through to text-only: we would need a separate text-only path;
            # for SKU extraction we fail fast instead.
            raise ModelNotVisionError(
                f"Model '{model}' does not support images and no vision fallback is configured. "
                f"Set OLLAMA_VISION_FALLBACK to a vision model (e.g. llava:7b)."
            )
        else:  # fail_fast
            raise ModelNotVisionError(
                f"Model '{model}' does not support images (text-only). "
                f"Please choose a vision-capable model (e.g. llava:7b, llama3.2-vision:latest). "
                f"Configured fallback: {fb or 'none'}."
            )

    user_prompt = USER_PROMPT_TEMPLATE

    # --- 2. Try OpenAI-compatible endpoint ---
    try:
        res = _ollama_via_openai(client, model, user_prompt, image_b64)
        return res
    except Exception as exc:  # noqa: BLE001
        # 2a. mllama architecture error (old Ollama) is NOT retryable — give clear upgrade hint
        if _is_mllama_error(exc):
            print(
                f"[llm] Request rejected: model '{model}' requires a newer Ollama (mllama architecture). "
                f"Please run: winget upgrade Ollama.Ollama and restart Ollama. Original: {exc}",
                flush=True,
            )
            # Try vision fallback if available and not already tried
            fb = _resolve_vision_fallback(model)
            if fb and fb.lower() != model.lower():
                print(f"[llm] Auto-switching to fallback '{fb}' due to mllama error.", flush=True)
                try:
                    res = _ollama_via_openai(client, fb, user_prompt, image_b64)
                    return res
                except Exception as fb_exc:
                    # If fallback also fails with mllama, give up
                    if _is_mllama_error(fb_exc):
                        raise ModelNotVisionError(f"Both '{model}' and fallback '{fb}' require newer Ollama (mllama). Please upgrade Ollama.") from fb_exc
                    raise
            raise ModelNotVisionError(f"Model '{model}' requires newer Ollama (mllama). Please upgrade Ollama.") from exc

        # 2b. For 400s, decide: retryable vs non-retryable
        if _is_ollama_400(exc):
            # If it's a vision model but OpenAI shim says "multimodal not supported",
            # that's a genuine endpoint difference (llava) -> try native ONCE.
            if is_vision_model(model) and _is_not_vision_error(exc):
                # This is the ONLY case where OpenAI->native fallback is useful
                base = str(getattr(client, "base_url", "") or "").replace("/v1", "").rstrip("/")
                if not base:
                    base = "http://localhost:11434"
                print(
                    f"[llm] OpenAI endpoint rejected vision input for '{model}' (endpoint limitation, not model); "
                    f"trying native /api/chat once.",
                    flush=True,
                )
                try:
                    res = _ollama_via_native(base, model, user_prompt, image_b64)
                    return res
                except Exception as native_exc:  # noqa: BLE001
                    # If native also says not vision, then model is text-only (should have been caught pre-flight)
                    # but handle gracefully with single clear line, no retry
                    if _is_ollama_400(native_exc) and _is_not_vision_error(native_exc):
                        print(
                            f"[llm] Request rejected: model '{model}' does not support image input. "
                            f"Configured vision fallback: {VISION_FALLBACK_MODEL if is_vision_model(VISION_FALLBACK_MODEL) else 'none'}.",
                            flush=True,
                        )
                        raise ModelNotVisionError(
                            f"Model '{model}' does not support images (text-only). "
                            f"Select a vision model like 'llava:7b' or 'llama3.2-vision:latest'."
                        ) from native_exc
                    # Other native 400 (bad params) -> fail fast, don't retry
                    if _is_ollama_400(native_exc):
                        print(
                            f"[llm] Request rejected: native endpoint 400 for '{model}' (bad request, not retryable): {native_exc}",
                            flush=True,
                        )
                        raise
                    raise
            # For all other 400s (invalid params, model not found, or text-only model that slipped through
            # pre-flight), fail fast with single clear line — don't fall back, don't retry
            print(
                f"[llm] Request rejected: model '{model}' 400 Bad Request (not retryable, not falling back): {exc}",
                flush=True,
            )
            raise

        # 2c. For retryable errors (5xx, connection, timeout), let tenacity retry
        raise


def _ollama_via_openai(client, model, user_prompt, image_b64):
    data_url = f"data:image/jpeg;base64,{image_b64}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        max_tokens=4096,  # match Claude path; small shelves can produce long JSON
        temperature=0.0,
    )
    if not getattr(response, "choices", None):
        raise ValueError(f"Empty choices in Ollama response for model '{model}'")
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError(f"Empty message content from Ollama model '{model}'")
    return _finalize(text)


def _ollama_via_native(base_url, model, user_prompt, image_b64):
    """Native Ollama /api/chat endpoint (images as raw base64)."""
    import json as _json
    import urllib.request

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt, "images": [image_b64]},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 4096},
    }
    data = _json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = resp.read().decode("utf-8")
    result = _json.loads(body)
    text = (result.get("message") or {}).get("content", "").strip()
    if not text:
        raise ValueError(f"Empty message content from native Ollama endpoint for model '{model}'")
    return _finalize(text)


def _finalize(text):
    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        raise ValueError("Top-level JSON is not an object")
    parsed.setdefault("item_count", 0)
    parsed.setdefault("items", [])
    return parsed

import google.generativeai as genai
import os

api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

import base64
from io import BytesIO
from PIL import Image

@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type(Exception),
)
def extract_with_gemini(
    model_name: str, image_b64: str
) -> dict:
    """Send the image to Google Gemini and return the parsed JSON."""
    user_prompt = USER_PROMPT_TEMPLATE
    
    # decode base64 to image
    image_data = base64.b64decode(image_b64)
    img = Image.open(BytesIO(image_data))
    
    model = genai.GenerativeModel(model_name=model_name, system_instruction=SYSTEM_PROMPT)
    response = model.generate_content(
        [user_prompt, img],
        generation_config=genai.GenerationConfig(temperature=0.0)
    )
    
    text = response.text
    return _finalize(text)
