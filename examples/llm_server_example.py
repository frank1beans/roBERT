#!/usr/bin/env python3
"""Simple mock LLM server for testing property extraction.

This server provides a minimal FastAPI endpoint that simulates an LLM
for property extraction, useful for testing without real API calls.
"""
import argparse
import json
import re
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


app = FastAPI(title="Mock LLM Server", version="0.1.0")


class ExtractionRequest(BaseModel):
    """Request schema for property extraction."""
    prompt: str
    schema: Dict[str, Any]
    model: str = "mock"


class ExtractionResponse(BaseModel):
    """Response schema for property extraction."""
    value: Any
    confidence: float = 0.85
    source: str = "llm_mock"


def mock_extract_property(text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Mock extraction logic using simple pattern matching.

    This is a simplified extractor for demonstration purposes.
    In production, this would call a real LLM API.
    """
    # Extract numbers with units (e.g., "12mm", "60x60")
    if "spessore" in text.lower() or "sp." in text.lower():
        match = re.search(r'(?:sp\.?\s*|spessore\s+)(\d+(?:[.,]\d+)?)\s*(mm|cm)?', text, re.IGNORECASE)
        if match:
            value = match.group(1).replace(',', '.')
            unit = match.group(2) or "mm"
            return {"value": f"{value}{unit}", "confidence": 0.9, "source": "llm_mock"}

    # Extract dimensions (e.g., "60x60", "120x60x10")
    if "dimension" in str(schema).lower():
        match = re.search(r'(\d+)\s*[xX×]\s*(\d+)(?:\s*[xX×]\s*(\d+))?', text)
        if match:
            if match.group(3):
                value = f"{match.group(1)}x{match.group(2)}x{match.group(3)}"
            else:
                value = f"{match.group(1)}x{match.group(2)}"
            return {"value": value, "confidence": 0.88, "source": "llm_mock"}

    # Extract materials
    materials = ["gres porcellanato", "ceramica", "legno", "metallo", "plastica",
                 "vetro", "cartongesso", "calcestruzzo", "acciaio"]
    for material in materials:
        if material in text.lower():
            return {"value": material, "confidence": 0.85, "source": "llm_mock"}

    # Fallback: return null
    return {"value": None, "confidence": 0.0, "source": "llm_mock"}


@app.post("/extract")
async def extract_property(request: ExtractionRequest) -> ExtractionResponse:
    """Extract property from text using mock LLM logic.

    This endpoint simulates an LLM extraction service. In production,
    replace this with actual API calls to GPT-4, Claude, etc.
    """
    try:
        # Extract text from prompt (assumes specific format)
        # Expected format: "Testo:\n<text>\nDomanda:\n<question>\nSchema:\n<schema>"
        lines = request.prompt.split('\n')
        text = ""
        for i, line in enumerate(lines):
            if line.startswith("Testo:") and i + 1 < len(lines):
                # Get all lines until next section
                j = i + 1
                while j < len(lines) and not lines[j].startswith("Domanda:"):
                    text += lines[j] + " "
                    j += 1
                break

        text = text.strip()
        if not text:
            raise ValueError("No text found in prompt")

        result = mock_extract_property(text, request.schema)

        return ExtractionResponse(
            value=result["value"],
            confidence=result["confidence"],
            source=result["source"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": "mock"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Mock LLM Server",
        "version": "0.1.0",
        "endpoints": {
            "extract": "POST /extract - Extract properties from text",
            "health": "GET /health - Health check"
        }
    }


def main():
    """Start the mock LLM server."""
    parser = argparse.ArgumentParser(description="Mock LLM server for property extraction")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print(f"🚀 Starting Mock LLM Server on http://{args.host}:{args.port}")
    print(f"📝 Endpoint: POST http://{args.host}:{args.port}/extract")
    print(f"❤️  Health: GET http://{args.host}:{args.port}/health")

    uvicorn.run(
        "llm_server_example:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
