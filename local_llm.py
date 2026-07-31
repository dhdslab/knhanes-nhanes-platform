# -*- coding: utf-8 -*-
"""Small local Ollama client used by the report factory.

This module deliberately uses only the Python standard library so the model
check can run in the bundled Codex Python environment without installing
requests, transformers, or llama-cpp-python.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


DEFAULT_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3.2:latest")
DEFAULT_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
DEFAULT_TIMEOUT = float(os.getenv("LOCAL_LLM_TIMEOUT", "300"))


def _clean_url(url: str | None) -> str:
    return (url or DEFAULT_URL).rstrip("/")


def _json_request(
    url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _clean_url(url) + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not connect to local Ollama. Start it with `ollama serve` "
            f"or check LOCAL_LLM_URL. Original error: {exc.reason}"
        ) from exc


def list_models(url: str | None = None, timeout: float = 10) -> list[str]:
    data = _json_request(_clean_url(url), "/api/tags", timeout=timeout)
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


def model_available(
    model: str | None = None,
    url: str | None = None,
    timeout: float = 10,
) -> bool:
    wanted = model or DEFAULT_MODEL
    names = list_models(url, timeout=timeout)
    return wanted in names or wanted.split(":", 1)[0] in {n.split(":", 1)[0] for n in names}


def generate(
    prompt: str,
    model: str | None = None,
    url: str | None = None,
    fmt: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    body: dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    if fmt:
        body["format"] = fmt

    temperature = os.getenv("LOCAL_LLM_TEMPERATURE")
    num_ctx = os.getenv("LOCAL_LLM_NUM_CTX")
    options: dict[str, Any] = {}
    if temperature:
        options["temperature"] = float(temperature)
    if num_ctx:
        options["num_ctx"] = int(num_ctx)
    if options:
        body["options"] = options

    data = _json_request(_clean_url(url), "/api/generate", body, timeout=timeout)
    if "error" in data:
        raise RuntimeError(f"Ollama error: {data['error']}")
    if "response" not in data:
        raise RuntimeError(f"Ollama returned an unexpected payload: {data}")
    return str(data["response"])
