# roBERT Makefile - Common tasks automation

# =============================================================================
# Configuration
# =============================================================================

# QA Model configuration
MODEL ?= atipiqal/BOB
OUT_DIR ?= outputs/qa_models/base
TRAIN_JSONL ?= data/tmp/qa_train.jsonl
EVAL_JSONL ?=
TEXT ?= "Parete in cartongesso spessore 12,5 mm"
CATEGORY ?= controsoffitti
REGISTRY ?= resources/data/properties/registry.json
NULL_TH ?= 0.25

# LLM configuration
LLM_PORT ?= 8000
LLM_ENDPOINT ?= http://localhost:$(LLM_PORT)/extract
LLM_MODEL ?= test
LLM_PROVIDER ?= mock

# Extraction configuration
INPUT ?= resources/data/train/classification/raw/dataset_lim.jsonl
OUTPUT ?= outputs/extracted.jsonl
SAMPLE ?= 20

.PHONY: help qa-train qa-predict qa-smoke tests
.PHONY: llm-server llm-test extract-rules extract-llm analyze
.PHONY: clean install

qa-train:
	python -m robimb.extraction.property_qa train \
	  --model $(MODEL) \
	  --train-jsonl $(TRAIN_JSONL) \
	  $(if $(EVAL_JSONL),--eval-jsonl $(EVAL_JSONL),) \
	  --out-dir $(OUT_DIR)

qa-predict:
	python -m robimb.extraction.property_qa predict \
	  --model-dir $(OUT_DIR) \
	  --text $(TEXT) \
	  --category $(CATEGORY) \
	  --registry $(REGISTRY) \
	  --null-th $(NULL_TH)

qa-smoke:
	MODEL=$(MODEL) OUT_DIR=$(OUT_DIR) TRAIN_JSONL=$(TRAIN_JSONL) $(MAKE) qa-train
	MODEL=$(MODEL) OUT_DIR=$(OUT_DIR) TEXT=$(TEXT) CATEGORY=$(CATEGORY) REGISTRY=$(REGISTRY) NULL_TH=$(NULL_TH) $(MAKE) qa-predict

tests:
	pytest -q

# =============================================================================
# LLM Endpoint Commands
# =============================================================================

llm-server:  ## Start LLM mock server
	@echo "Starting LLM $(LLM_PROVIDER) server on port $(LLM_PORT)..."
	@if [ "$(LLM_PROVIDER)" = "mock" ]; then \
		python examples/llm_server_example.py --port $(LLM_PORT); \
	else \
		python examples/llm_integration_examples.py $(LLM_PROVIDER); \
	fi

llm-test:  ## Test LLM server connection
	@echo "Testing LLM endpoint at $(LLM_ENDPOINT)..."
	@curl -s -X POST $(LLM_ENDPOINT) \
		-H "Content-Type: application/json" \
		-d '{"prompt":"Testo:\nLastra sp. 12mm\nDomanda:\nEstrai spessore\nSchema:\n{}","schema":{},"model":"$(LLM_MODEL)"}' | \
		python -m json.tool || echo "ERROR: LLM server not responding"

llm-integration-test:  ## Run full LLM integration test
	@bash scripts/testing/test_llm_integration.sh

# =============================================================================
# Extraction Commands
# =============================================================================

extract-rules:  ## Extract properties using rules only (no LLM, no QA)
	@echo "Extracting properties with rules only..."
	robimb extract properties \
		--input $(INPUT) \
		--output outputs/rules_only.jsonl \
		--no-qa \
		--sample $(SAMPLE)
	@$(MAKE) analyze OUTPUT=outputs/rules_only.jsonl

extract-llm:  ## Extract properties using rules + LLM
	@echo "Extracting properties with LLM endpoint..."
	robimb extract properties \
		--input $(INPUT) \
		--output outputs/with_llm.jsonl \
		--llm-endpoint $(LLM_ENDPOINT) \
		--llm-model $(LLM_MODEL) \
		--no-qa \
		--sample $(SAMPLE)
	@$(MAKE) analyze OUTPUT=outputs/with_llm.jsonl

extract-full:  ## Extract properties using rules + QA + LLM
	@echo "Extracting properties with full pipeline..."
	robimb extract properties \
		--input $(INPUT) \
		--output outputs/full_pipeline.jsonl \
		--llm-endpoint $(LLM_ENDPOINT) \
		--llm-model $(LLM_MODEL) \
		--use-qa \
		--qa-model-dir $(OUT_DIR) \
		--sample $(SAMPLE)
	@$(MAKE) analyze OUTPUT=outputs/full_pipeline.jsonl

analyze:  ## Analyze extraction results
	@echo "Analyzing results from $(OUTPUT)..."
	@python scripts/analysis/extraction_results.py $(OUTPUT)

# =============================================================================
# Comparison Commands
# =============================================================================

compare-llm:  ## Compare rules-only vs with-LLM extraction
	@echo "Running comparison: rules-only vs with-LLM..."
	@$(MAKE) extract-rules SAMPLE=$(SAMPLE)
	@$(MAKE) extract-llm SAMPLE=$(SAMPLE)
	@echo ""
	@echo "========================================="
	@echo "COMPARISON RESULTS"
	@echo "========================================="
	@echo ""
	@echo "--- RULES ONLY ---"
	@python scripts/analysis/extraction_results.py outputs/rules_only.jsonl | head -20
	@echo ""
	@echo "--- WITH LLM ---"
	@python scripts/analysis/extraction_results.py outputs/with_llm.jsonl | head -20

# =============================================================================
# Setup and Maintenance
# =============================================================================

install:  ## Install dependencies
	pip install -e ".[dev]"

clean:  ## Clean output files and cache
	rm -rf outputs/*.jsonl
	rm -rf .llm_cache
	rm -rf __pycache__ **/__pycache__
	rm -rf .pytest_cache
	find . -name "*.pyc" -delete

help:  ## Show this help message
	@echo "roBERT Makefile - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make llm-server                    # Start mock LLM server"
	@echo "  make extract-llm SAMPLE=50         # Extract 50 docs with LLM"
	@echo "  make compare-llm SAMPLE=100        # Compare rules vs LLM"
	@echo "  make llm-server LLM_PROVIDER=openai  # Use OpenAI instead of mock"
	@echo ""

.DEFAULT_GOAL := help
