"""OpenAI image generation backend.

Exposes OpenAI's ``gpt-image-2`` model at three quality tiers as an
:class:`ImageGenProvider` implementation. The tiers are implemented as
three virtual model IDs so the ``hermes tools`` model picker and the
``image_gen.model`` config key behave like any other multi-model backend:

    gpt-image-2-low     ~15s   fastest, good for iteration
    gpt-image-2-medium  ~40s   default — balanced
    gpt-image-2-high    ~2min  slowest, highest fidelity

All three hit the same underlying API model (``gpt-image-2``) with a
different ``quality`` parameter. Output is base64 JSON → saved under
``$HERMES_HOME/cache/images/``.

Selection precedence (first hit wins):

1. ``OPENAI_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.openai.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it's one of our tier IDs)
4. :data:`DEFAULT_MODEL` — ``gpt-image-2-medium``
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from agent.secret_scope import get_secret
from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
#
# All three IDs resolve to the same underlying API model with a different
# ``quality`` setting. ``api_model`` is what gets sent to OpenAI;
# ``quality`` is the knob that changes generation time and output fidelity.

API_MODEL = "gpt-image-2"

_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-image-2-low": {
        "display": "GPT Image 2 (Low)",
        "speed": "~15s",
        "strengths": "Fast iteration, lowest cost",
        "quality": "low",
    },
    "gpt-image-2-medium": {
        "display": "GPT Image 2 (Medium)",
        "speed": "~40s",
        "strengths": "Balanced — default",
        "quality": "medium",
    },
    "gpt-image-2-high": {
        "display": "GPT Image 2 (High)",
        "speed": "~2min",
        "strengths": "Highest fidelity, strongest prompt adherence",
        "quality": "high",
    },
}

DEFAULT_MODEL = "gpt-image-2-medium"

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}

_MAX_INPUT_IMAGE_BYTES = 25 * 1024 * 1024
_ACCEPTED_INPUT_MIME = frozenset({"image/png", "image/jpeg", "image/webp"})


def _load_openai_config() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which tier to use and return ``(model_id, meta)``."""
    env_override = os.environ.get("OPENAI_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_openai_config()
    openai_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(openai_cfg, dict):
        value = openai_cfg.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# Source-image loading (for image-to-image / edit)
# ---------------------------------------------------------------------------


def _validated_image_bytes(data: bytes, filename: str) -> Tuple[bytes, str]:
    """Enforce edit-input size and raster magic bytes for every source shape."""
    if not data:
        raise ValueError("Image input is empty")
    if len(data) > _MAX_INPUT_IMAGE_BYTES:
        raise ValueError("Image input exceeds 25MB cap")

    from agent.image_routing import _sniff_mime_from_bytes

    mime = _sniff_mime_from_bytes(data)
    if mime not in _ACCEPTED_INPUT_MIME:
        raise ValueError("Image input does not contain supported image bytes")
    extension = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
    }[mime]
    stem, _current_extension = os.path.splitext(filename or "image")
    safe_name = f"{stem or 'image'}{extension}"
    return data, safe_name


def _load_image_bytes(ref: str) -> Tuple[bytes, str]:
    """Load and validate image bytes from a URL, data URL, or local path."""
    ref = ref.strip()
    lower = ref.lower()
    if lower.startswith(("http://", "https://")):
        from tools.url_safety import (
            create_ssrf_safe_client,
            is_safe_url,
            redirect_target_from_response,
        )

        if not is_safe_url(ref):
            raise ValueError("Blocked unsafe URL (SSRF protection)")

        def _redirect_guard(response: Any) -> None:
            target = redirect_target_from_response(response)
            if target and not is_safe_url(target):
                raise ValueError("Blocked unsafe redirect (SSRF protection)")

        with create_ssrf_safe_client(
            timeout=60.0,
            follow_redirects=True,
            event_hooks={"response": [_redirect_guard]},
        ) as client:
            with client.stream("GET", ref) as resp:
                resp.raise_for_status()
                length_header = resp.headers.get("Content-Length")
                if length_header:
                    try:
                        declared_length = int(length_header)
                    except (TypeError, ValueError):
                        declared_length = 0
                    if declared_length > _MAX_INPUT_IMAGE_BYTES:
                        raise ValueError("Remote image exceeds 25MB cap")

                chunks: List[bytes] = []
                total = 0
                for chunk in resp.iter_bytes():
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > _MAX_INPUT_IMAGE_BYTES:
                        raise ValueError("Remote image exceeds 25MB cap")
                    chunks.append(chunk)
                data = b"".join(chunks)

        path_name = os.path.basename(urlparse(ref).path) or "image.png"
        return _validated_image_bytes(data, path_name)

    if lower.startswith("data:"):
        import base64

        header, separator, b64 = ref.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError("Image data URL must be base64 encoded")
        ext = "png"
        if "image/" in header:
            ext = header.split("image/", 1)[1].split(";", 1)[0] or "png"
        try:
            data = base64.b64decode(b64, validate=True)
        except Exception as exc:
            raise ValueError("Image data URL is not valid base64") from exc
        return _validated_image_bytes(data, f"image.{ext}")

    # Local file path — enforce the shared credential-read guard before reading.
    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(ref)
    with open(ref, "rb") as fh:
        data = fh.read(_MAX_INPUT_IMAGE_BYTES + 1)
    name = os.path.basename(ref) or "image.png"
    return _validated_image_bytes(data, name)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAIImageGenProvider(ImageGenProvider):
    """OpenAI ``images.generate`` / ``images.edit`` backend — gpt-image-2."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI"

    def is_available(self) -> bool:
        if not get_secret("OPENAI_API_KEY"):
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "varies",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI",
            "badge": "paid",
            "tag": "gpt-image-2 at low/medium/high quality tiers — text-to-image & image editing",
            "env_vars": [
                {
                    "key": "OPENAI_API_KEY",
                    "prompt": "OpenAI API key",
                    "url": "https://platform.openai.com/api-keys",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        # gpt-image-2 supports editing via images.edit() with up to 16 source
        # images.
        return {"modalities": ["text", "image"], "max_reference_images": 16}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="openai",
                aspect_ratio=aspect,
            )

        api_key = get_secret("OPENAI_API_KEY")
        if not api_key:
            return error_response(
                error=(
                    "OPENAI_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → OpenAI to configure, or `hermes setup` "
                    "to add the key."
                ),
                error_type="auth_required",
                provider="openai",
                aspect_ratio=aspect,
            )

        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider="openai",
                aspect_ratio=aspect,
            )

        tier_id, meta = _resolve_model()
        size = _SIZES.get(aspect, _SIZES["square"])

        # Collect source images (primary + references) for image-to-image.
        sources: List[str] = []
        if isinstance(image_url, str) and image_url.strip():
            sources.append(image_url.strip())
        for ref in (normalize_reference_images(reference_image_urls) or []):
            sources.append(ref)
        sources = sources[:16]  # gpt-image-2 edit caps at 16 images
        is_edit = bool(sources)
        modality = "image" if is_edit else "text"

        client = openai.OpenAI(api_key=api_key)

        if is_edit:
            # images.edit() expects file-like objects. Download/read each
            # source into a named BytesIO so the SDK sends correct multipart.
            import io

            try:
                files = []
                for ref in sources:
                    data, fname = _load_image_bytes(ref)
                    bio = io.BytesIO(data)
                    bio.name = fname
                    files.append(bio)
            except Exception as exc:
                return error_response(
                    error=f"Could not load source image for editing: {exc}",
                    error_type="io_error",
                    provider="openai",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            try:
                response = client.images.edit(
                    model=API_MODEL,
                    image=files if len(files) > 1 else files[0],
                    prompt=prompt,
                    size=size,  # type: ignore[arg-type]  # _SIZES values are valid gpt-image sizes
                    quality=meta["quality"],
                    n=1,
                )
            except Exception as exc:
                logger.debug("OpenAI image edit failed", exc_info=True)
                return error_response(
                    error=f"OpenAI image editing failed: {exc}",
                    error_type="api_error",
                    provider="openai",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        else:
            # gpt-image-2 returns b64_json unconditionally and REJECTS
            # ``response_format`` as an unknown parameter. Don't send it.
            payload: Dict[str, Any] = {
                "model": API_MODEL,
                "prompt": prompt,
                "size": size,
                "n": 1,
                "quality": meta["quality"],
            }

            try:
                response = client.images.generate(**payload)
            except Exception as exc:
                logger.debug("OpenAI image generation failed", exc_info=True)
                return error_response(
                    error=f"OpenAI image generation failed: {exc}",
                    error_type="api_error",
                    provider="openai",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

        data = getattr(response, "data", None) or []
        if not data:
            return error_response(
                error="OpenAI returned no image data",
                error_type="empty_response",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = data[0]
        b64 = getattr(first, "b64_json", None)
        url = getattr(first, "url", None)
        revised_prompt = getattr(first, "revised_prompt", None)

        if b64:
            try:
                saved_path = save_b64_image(b64, prefix=f"openai_{tier_id}")
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider="openai",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        elif url:
            # Defensive — gpt-image-2 returns b64 today, but OpenAI's API
            # has previously returned URLs.  Cache the bytes locally so the
            # gateway never tries to fetch an ephemeral / signed URL after
            # it expires — same rationale as the xAI provider (#26942).
            try:
                saved_path = save_url_image(url, prefix=f"openai_{tier_id}")
            except Exception as exc:
                logger.warning(
                    "OpenAI image URL %s could not be cached (%s); falling back to bare URL.",
                    url,
                    exc,
                )
                image_ref = url
            else:
                image_ref = str(saved_path)
        else:
            return error_response(
                error="OpenAI response contained neither b64_json nor URL",
                error_type="empty_response",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {"size": size, "quality": meta["quality"]}
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt

        return success_response(
            image=image_ref,
            model=tier_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="openai",
            modality=modality,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``OpenAIImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(OpenAIImageGenProvider())
