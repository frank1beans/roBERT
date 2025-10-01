#!/bin/bash
# Script per riorganizzare i file del progetto roBERT

set -e

echo "=========================================="
echo "Riorganizzazione File roBERT"
echo "=========================================="
echo ""

# 1. Documentazione → docs/
echo "[1/5] Spostando documentazione in docs/..."
mv SETUP_GPT4MINI.md docs/ 2>/dev/null || echo "  SETUP_GPT4MINI.md già in docs/"
mv GPT4MINI_READY.md docs/ 2>/dev/null || echo "  GPT4MINI_READY.md già in docs/"
mv LLM_QUICKSTART.md docs/ 2>/dev/null || echo "  LLM_QUICKSTART.md già in docs/"
mv IMPLEMENTATION_SUMMARY.md docs/ 2>/dev/null || echo "  IMPLEMENTATION_SUMMARY.md già in docs/"
mv PERFORMANCE_OPTIMIZATION.md docs/ 2>/dev/null || echo "  PERFORMANCE_OPTIMIZATION.md già in docs/"
mv FINAL_SUMMARY.md docs/ 2>/dev/null || echo "  FINAL_SUMMARY.md già in docs/"
echo "  ✓ Documentazione spostata"

# 2. Server/Esempi → examples/
echo ""
echo "[2/5] Spostando server ed esempi in examples/..."
mv llm_server_example.py examples/ 2>/dev/null || echo "  llm_server_example.py già in examples/"
mv llm_server_gpt4mini.py examples/ 2>/dev/null || echo "  llm_server_gpt4mini.py già in examples/"
echo "  ✓ Server/esempi spostati"

# 3. Script → scripts/
echo ""
echo "[3/5] Spostando script in scripts/..."
mv analyze_extraction.py scripts/ 2>/dev/null || echo "  analyze_extraction.py già in scripts/"
mv test_llm_integration.sh scripts/ 2>/dev/null || echo "  test_llm_integration.sh già in scripts/"
mv setup_gpt4mini.ps1 scripts/ 2>/dev/null || echo "  setup_gpt4mini.ps1 già in scripts/"
echo "  ✓ Script spostati"

# 4. Pulizia file temporanei
echo ""
echo "[4/5] Pulendo file temporanei..."
rm -f temp_*.jsonl 2>/dev/null || true
rm -f batch_*.jsonl 2>/dev/null || true
rm -f PREAMBOLO.md 2>/dev/null || true
echo "  ✓ File temporanei eliminati"

# 5. Aggiorna .gitignore
echo ""
echo "[5/5] Aggiornando .gitignore..."
cat >> .gitignore << 'EOF'

# Environment files
.env
.env.*
*.env

# LLM cache
.llm_cache/

# Temporary files
temp_*.jsonl
batch_*.jsonl

# Outputs (keep folder but ignore content)
outputs/*.jsonl
!outputs/.gitkeep
EOF
echo "  ✓ .gitignore aggiornato"

echo ""
echo "=========================================="
echo "✓ Riorganizzazione Completata!"
echo "=========================================="
echo ""
echo "Prossimi passi:"
echo "  1. Verifica che i file siano nel posto giusto: ls docs/ examples/ scripts/"
echo "  2. Testa i comandi: make help"
echo "  3. Aggiorna README.md con i nuovi path"
echo ""
