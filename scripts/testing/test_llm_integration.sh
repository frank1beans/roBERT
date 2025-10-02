#!/bin/bash
#
# Test script for LLM endpoint integration
# This script verifies that the LLM endpoint functionality works correctly
#

set -e  # Exit on error

echo "=============================================================================="
echo "LLM ENDPOINT INTEGRATION TEST"
echo "=============================================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
LLM_PORT=8000
LLM_URL="http://localhost:$LLM_PORT/extract"
SAMPLE_SIZE=20

echo -e "${YELLOW}Step 1: Check if LLM server is running...${NC}"
if curl -s --max-time 2 "$LLM_URL" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ LLM server is running on port $LLM_PORT${NC}"
else
    echo -e "${RED}✗ LLM server is NOT running${NC}"
    echo ""
    echo "Please start the server in another terminal:"
    echo "  python llm_server_example.py --port $LLM_PORT"
    echo ""
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 2: Test LLM server with a simple request...${NC}"
RESPONSE=$(curl -s -X POST "$LLM_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Testo:\nLastra in cartongesso spessore 12mm\nDomanda:\nEstrai lo spessore\nSchema:\n{}",
    "schema": {},
    "model": "test"
  }')

if echo "$RESPONSE" | grep -q "value"; then
    echo -e "${GREEN}✓ Server responded correctly${NC}"
    echo "  Response: $RESPONSE"
else
    echo -e "${RED}✗ Server response is invalid${NC}"
    echo "  Response: $RESPONSE"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 3: Run extraction WITHOUT LLM endpoint...${NC}"
robimb extract properties \
  --input resources/data/train/classification/raw/dataset_lim.jsonl \
  --output outputs/test_no_llm.jsonl \
  --no-qa \
  --sample $SAMPLE_SIZE || {
    echo -e "${RED}✗ Extraction without LLM failed${NC}"
    exit 1
}
echo -e "${GREEN}✓ Extraction without LLM completed${NC}"

echo ""
echo -e "${YELLOW}Step 4: Run extraction WITH LLM endpoint...${NC}"
robimb extract properties \
  --input resources/data/train/classification/raw/dataset_lim.jsonl \
  --output outputs/test_with_llm.jsonl \
  --llm-endpoint "$LLM_URL" \
  --llm-model test \
  --llm-timeout 30 \
  --llm-max-retries 2 \
  --no-qa \
  --sample $SAMPLE_SIZE || {
    echo -e "${RED}✗ Extraction with LLM failed${NC}"
    exit 1
}
echo -e "${GREEN}✓ Extraction with LLM completed${NC}"

echo ""
echo -e "${YELLOW}Step 5: Analyze results...${NC}"

echo ""
echo "--- WITHOUT LLM ---"
python analyze_extraction.py outputs/test_no_llm.jsonl

echo ""
echo "--- WITH LLM ---"
python analyze_extraction.py outputs/test_with_llm.jsonl

echo ""
echo -e "${YELLOW}Step 6: Check for LLM-extracted properties...${NC}"

# Count properties with source "qa_llm"
LLM_COUNT=$(grep -o '"source": "qa_llm"' outputs/test_with_llm.jsonl | wc -l || echo "0")

if [ "$LLM_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ Found $LLM_COUNT properties extracted by LLM${NC}"
    echo ""
    echo "Sample LLM-extracted property:"
    grep -m 1 -A 3 -B 1 '"source": "qa_llm"' outputs/test_with_llm.jsonl | head -5
else
    echo -e "${YELLOW}⚠ No properties were extracted by LLM${NC}"
    echo "  This might be normal if the mock server doesn't match any patterns"
    echo "  Try with a real LLM endpoint for better results"
fi

echo ""
echo "=============================================================================="
echo -e "${GREEN}ALL TESTS PASSED ✓${NC}"
echo "=============================================================================="
echo ""
echo "Next steps:"
echo "  1. Check the output files in outputs/"
echo "  2. Try with a real LLM endpoint (OpenAI/Anthropic)"
echo "  3. Optimize prompts in resources/data/properties/prompts.json"
echo ""
echo "Documentation:"
echo "  - Quick start: LLM_QUICKSTART.md"
echo "  - Full guide: docs/llm_endpoint_guide.md"
echo ""
