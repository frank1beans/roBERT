#!/usr/bin/env python3
"""LLM server using OpenAI GPT-4o-mini for property extraction.

This server provides a FastAPI endpoint that uses OpenAI's GPT-4o-mini
for actual property extraction from BIM descriptions.
"""
import argparse
import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "OpenAI package not installed. Install with: pip install openai"
    )


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GPT-4o-mini LLM Server",
    description="Property extraction using OpenAI GPT-4o-mini",
    version="0.1.0"
)

# Global OpenAI client
client: Optional[OpenAI] = None


class ExtractionRequest(BaseModel):
    """Request schema for property extraction."""
    prompt: str
    schema: Dict[str, Any]
    model: str = "gpt-4o-mini"


class ExtractionResponse(BaseModel):
    """Response schema for property extraction."""
    value: Any
    confidence: float = 0.85
    source: str = "llm"
    raw_response: Optional[str] = None


def extract_with_gpt4mini(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 500
) -> Dict[str, Any]:
    """Extract property using GPT-4o-mini.

    Args:
        prompt: The formatted prompt with text, question, and schema
        model: OpenAI model name (default: gpt-4o-mini)
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum tokens in response

    Returns:
        Dictionary with extracted value and metadata
    """
    if not client:
        raise RuntimeError("OpenAI client not initialized. Check API key.")

    try:
        # Add JSON formatting instruction
        system_prompt = (
            "Sei un assistente esperto nell'estrazione di proprietà da descrizioni BIM. "
            "Rispondi SEMPRE con un JSON valido nel formato: "
            '{"value": <valore_estratto>, "confidence": <0.0-1.0>}. '
            "Se non trovi la proprietà, usa: "
            '{"value": null, "confidence": 0.0}'
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}  # Force JSON output
        )

        raw_content = response.choices[0].message.content
        logger.info(f"GPT-4o-mini response: {raw_content}")

        # Parse JSON response
        try:
            result = json.loads(raw_content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON: {raw_content}")
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = {"value": None, "confidence": 0.0}

        # Ensure required fields
        if "value" not in result:
            result["value"] = None
        if "confidence" not in result:
            result["confidence"] = 0.85 if result["value"] else 0.0

        result["raw_response"] = raw_content
        result["source"] = "gpt-4o-mini"

        return result

    except Exception as e:
        logger.error(f"Error calling GPT-4o-mini: {e}", exc_info=True)
        raise


@app.post("/extract")
async def extract_property(request: ExtractionRequest) -> ExtractionResponse:
    """Extract property from text using GPT-4o-mini.

    Expects a prompt in the format:
        Testo:
        <description>
        Domanda:
        <question>
        Schema:
        <json_schema>
    """
    try:
        result = extract_with_gpt4mini(
            prompt=request.prompt,
            model=request.model or "gpt-4o-mini"
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
    api_key_configured = bool(client)
    return {
        "status": "healthy" if api_key_configured else "api_key_missing",
        "model": "gpt-4o-mini",
        "api_key_configured": api_key_configured
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "GPT-4o-mini LLM Server",
        "version": "0.1.0",
        "model": "gpt-4o-mini",
        "endpoints": {
            "extract": "POST /extract - Extract properties from text",
            "health": "GET /health - Health check"
        }
    }


@app.on_event("startup")
async def startup_event():
    """Initialize OpenAI client on startup."""
    global client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "❌ OPENAI_API_KEY not found in environment variables!\n"
            "   Set it with: export OPENAI_API_KEY='your-key-here'\n"
            "   Or add to .env file"
        )
        # Don't raise, allow server to start but fail on requests
        client = None
    else:
        client = OpenAI(api_key=api_key)
        logger.info("✅ OpenAI client initialized successfully")


def main():
    """Start the GPT-4o-mini LLM server."""
    parser = argparse.ArgumentParser(
        description="GPT-4o-mini LLM server for property extraction"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    # Check API key before starting
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

    print(f"🚀 Starting GPT-4o-mini LLM Server on http://{args.host}:{args.port}")
    print(f"📝 Endpoint: POST http://{args.host}:{args.port}/extract")
    print(f"❤️  Health: GET http://{args.host}:{args.port}/health")
    print()

    uvicorn.run(
        "llm_server_gpt4mini:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
