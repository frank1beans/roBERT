#!/usr/bin/env python3
"""LLM server for property extraction (OpenAI or Ollama).

This server exposes a FastAPI endpoint used by the pipeline to obtain
property values from a language model. It supports both OpenAI's
``chat.completions`` API (e.g. ``gpt-4o-mini``) and local Ollama models
such as ``llama3``.
"""
import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="robimb LLM Server",
    description="Property extraction using OpenAI or Ollama backends",
    version="0.1.0",
)

@dataclass
class ServerConfig:
    """Runtime configuration for the LLM server."""

    backend: str = "openai"
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3"
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 500


# Global configuration + optional client (for OpenAI backend)
config: Optional[ServerConfig] = None
client: Optional[Any] = None


class ExtractionRequest(BaseModel):
    """Request schema for property extraction."""

    prompt: str
    schema: Dict[str, Any]
    model: Optional[str] = None


class ExtractionResponse(BaseModel):
    """Response schema for property extraction."""
    value: Any
    confidence: float = 0.85
    source: str = "llm"
    raw_response: Optional[str] = None


JSON_INSTRUCTION = (
    "Sei un assistente esperto nell'estrazione di proprietà da descrizioni BIM. "
    "Rispondi SEMPRE con un JSON valido nel formato: "
    "{\"value\": <valore_estratto>, \"confidence\": <0.0-1.0>}. "
    "Se non trovi la proprietà, usa: {\"value\": null, \"confidence\": 0.0}"
)


def _require_config() -> ServerConfig:
    global config
    if config is None:
        config = _build_config_from_env()
    return config


def _build_config_from_env() -> ServerConfig:
    backend = os.getenv("LLM_BACKEND", "openai").lower()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def _parse_float(value: str, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return default

    def _parse_int(value: str, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return default

    temperature = _parse_float(os.getenv("LLM_TEMPERATURE", "0.0"), 0.0)
    max_tokens = _parse_int(os.getenv("LLM_MAX_TOKENS", "500"), 500)

    return ServerConfig(
        backend=backend,
        openai_model=openai_model,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _parse_model_response(raw_content: str) -> Dict[str, Any]:
    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON: {raw_content}")
        import re

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_content, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
        else:
            result = {"value": None, "confidence": 0.0}

    if "value" not in result:
        result["value"] = None
    if "confidence" not in result:
        result["confidence"] = 0.85 if result["value"] else 0.0
    return result


def extract_with_llm(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = _require_config()
    resolved_temperature = temperature if temperature is not None else cfg.temperature
    resolved_max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens

    if cfg.backend == "openai":
        if OpenAI is None:
            raise RuntimeError("openai package not installed. Install with: pip install openai")
        if client is None:
            raise RuntimeError("OpenAI client not initialized. Check OPENAI_API_KEY.")

        target_model = model or cfg.openai_model
        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": JSON_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content
        logger.info(f"OpenAI response: {raw_content}")
        result = _parse_model_response(raw_content)
        result["raw_response"] = raw_content
        result["source"] = target_model
        return result

    if cfg.backend == "ollama":
        # Map OpenAI model names to Ollama model
        if model and model.startswith("gpt-"):
            logger.warning(f"OpenAI model '{model}' requested but using Ollama backend, using '{cfg.ollama_model}' instead")
            target_model = cfg.ollama_model
        else:
            target_model = model or cfg.ollama_model
        host = cfg.ollama_host.rstrip("/")
        payload = {
            "model": target_model,
            "prompt": f"{JSON_INSTRUCTION}\n\n{prompt}\n",
            "stream": False,
            "options": {
                "temperature": resolved_temperature,
                "num_predict": resolved_max_tokens,
            },
        }

        url = f"{host}/api/generate"
        logger.info(f"Calling Ollama at: {url} with model: {target_model}")
        response = requests.post(
            url,
            json=payload,
            timeout=120,
        )
        logger.info(f"Response status: {response.status_code}, body: {response.text[:200]}")
        response.raise_for_status()
        data = response.json()
        raw_content = (data.get("response") or "").strip()
        logger.info(f"Ollama response: {raw_content}")
        result = _parse_model_response(raw_content)
        result["raw_response"] = raw_content
        result["source"] = target_model
        return result

    raise ValueError(f"Unsupported backend '{cfg.backend}'")


@app.post("/extract")
async def extract_property(request: ExtractionRequest) -> ExtractionResponse:
    """Extract property from text using the configured backend.

    Expects a prompt in the format:
        Testo:
        <description>
        Domanda:
        <question>
        Schema:
        <json_schema>
    """
    try:
        cfg = _require_config()
        default_model = cfg.openai_model if cfg.backend == "openai" else cfg.ollama_model
        result = extract_with_llm(
            prompt=request.prompt,
            model=request.model or default_model,
        )

        return ExtractionResponse(
            value=result["value"],
            confidence=result.get("confidence", 0.85),
            source=result.get("source", "llm"),
            raw_response=result.get("raw_response")
        )

    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    cfg = _require_config()
    if cfg.backend == "openai":
        api_key_configured = client is not None
    else:
        api_key_configured = True
    return {
        "status": "healthy" if api_key_configured else "api_key_missing",
        "backend": cfg.backend,
        "default_model": cfg.openai_model if cfg.backend == "openai" else cfg.ollama_model,
        "api_key_configured": api_key_configured,
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    cfg = _require_config()
    return {
        "service": "robimb LLM Server",
        "version": "0.1.0",
        "backend": cfg.backend,
        "default_model": cfg.openai_model if cfg.backend == "openai" else cfg.ollama_model,
        "endpoints": {
            "extract": "POST /extract - Extract properties from text",
            "health": "GET /health - Health check"
        }
    }


@app.on_event("startup")
async def startup_event():
    """Initialize OpenAI client on startup."""
    global client
    cfg = _require_config()

    if cfg.backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error(
                "❌ OPENAI_API_KEY not found in environment variables!\n"
                "   Set it with: export OPENAI_API_KEY='your-key-here'\n"
                "   Or add to .env file"
            )
            client = None
        else:
            client = OpenAI(api_key=api_key)
            logger.info("✅ OpenAI client initialized successfully")
    else:
        client = None  # OpenAI client not used
        host = cfg.ollama_host.rstrip("/")
        try:
            response = requests.get(f"{host}/api/tags", timeout=5)
            if response.status_code != 200:
                logger.warning(
                    "Ollama host responded with status %s when fetching tags.",
                    response.status_code,
                )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Unable to reach Ollama host %s (%s)", host, exc)


def main():
    """Start the GPT-4o-mini LLM server."""
    parser = argparse.ArgumentParser(
        description="robimb LLM server for property extraction"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument(
        "--backend",
        choices=["openai", "ollama"],
        default=os.getenv("LLM_BACKEND", "openai"),
        help="LLM backend to use (default: openai)",
    )
    parser.add_argument(
        "--openai-model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model name",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.getenv("OLLAMA_MODEL", "llama3"),
        help="Ollama model to invoke",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Base URL of the Ollama server",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("LLM_MAX_TOKENS", "500")),
        help="Maximum tokens to generate",
    )

    args = parser.parse_args()

    global config
    config = ServerConfig(
        backend=args.backend,
        openai_model=args.openai_model,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Persist configuration in environment variables so that reloads or
    # external imports reconstruct the same settings.
    os.environ["LLM_BACKEND"] = config.backend
    os.environ["OPENAI_MODEL"] = config.openai_model
    os.environ["OLLAMA_MODEL"] = config.ollama_model
    os.environ["OLLAMA_HOST"] = config.ollama_host
    os.environ["LLM_TEMPERATURE"] = str(config.temperature)
    os.environ["LLM_MAX_TOKENS"] = str(config.max_tokens)

    if args.backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  WARNING: OPENAI_API_KEY not set!")
            print("   Server will start but API calls will fail.")
            print()
            print("   To fix:")
            print("   1. Get API key from: https://platform.openai.com/api-keys")
            print("   2. Set environment variable:")
            print("      export OPENAI_API_KEY='sk-...'")
            print("   3. Or add to .env file in project root:")
            print("      OPENAI_API_KEY=sk-...")
            print()
        else:
            print(f"✅ OPENAI_API_KEY configured (starts with: {api_key[:10]}...)")
    else:
        print(f"🦙 Using Ollama backend at {args.ollama_host} (model: {args.ollama_model})")

    print(f"🚀 Starting robimb LLM Server on http://{args.host}:{args.port}")
    print(f"📝 Endpoint: POST http://{args.host}:{args.port}/extract")
    print(f"❤️  Health: GET http://{args.host}:{args.port}/health")
    print()

    target = app if not args.reload else "llm_server_gpt4mini:app"
    uvicorn.run(
        target,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
