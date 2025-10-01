# 📁 Piano di Riorganizzazione File

## 🎯 Obiettivo
Pulire la root e organizzare i file nelle cartelle appropriate.

---

## 📂 Struttura Target

```
roBERT/
├── README.md                              # Mantieni in root
├── Makefile                               # Mantieni in root
├── pyproject.toml                         # Mantieni in root
│
├── docs/                                  # Documentazione
│   ├── llm_endpoint_guide.md             # Già presente
│   ├── recommended_models.md             # Già presente
│   ├── cli_extract.md                    # Già presente
│   ├── SETUP_GPT4MINI.md                 # DA SPOSTARE
│   ├── GPT4MINI_READY.md                 # DA SPOSTARE
│   ├── LLM_QUICKSTART.md                 # DA SPOSTARE
│   ├── IMPLEMENTATION_SUMMARY.md         # DA SPOSTARE
│   ├── PERFORMANCE_OPTIMIZATION.md       # DA SPOSTARE
│   └── FINAL_SUMMARY.md                  # DA SPOSTARE
│
├── examples/                              # Esempi e server
│   ├── llm_integration_examples.py       # Già presente
│   ├── llm_server_example.py             # DA SPOSTARE
│   └── llm_server_gpt4mini.py            # DA SPOSTARE
│
├── scripts/                               # Script di utilità
│   ├── analysis/                         # Già presente
│   ├── analyze_extraction.py             # DA SPOSTARE
│   ├── test_llm_integration.sh           # DA SPOSTARE
│   └── setup_gpt4mini.ps1                # DA SPOSTARE
│
├── src/robimb/                            # Codice sorgente
├── tests/                                 # Test
├── outputs/                               # Output (gitignored)
└── resources/                             # Resources
```

---

## 🔄 Azioni da Eseguire

### 1. Spostare Documentazione → docs/
```bash
mv SETUP_GPT4MINI.md docs/
mv GPT4MINI_READY.md docs/
mv LLM_QUICKSTART.md docs/
mv IMPLEMENTATION_SUMMARY.md docs/
mv PERFORMANCE_OPTIMIZATION.md docs/
mv FINAL_SUMMARY.md docs/
```

### 2. Spostare Server/Esempi → examples/
```bash
mv llm_server_example.py examples/
mv llm_server_gpt4mini.py examples/
```

### 3. Spostare Script → scripts/
```bash
mv analyze_extraction.py scripts/
mv test_llm_integration.sh scripts/
mv setup_gpt4mini.ps1 scripts/
```

### 4. File da MANTENERE in Root
- ✅ README.md
- ✅ Makefile
- ✅ pyproject.toml
- ✅ .gitignore
- ✅ .env (gitignored)

### 5. File da ELIMINARE (se presenti)
- ❌ PREAMBOLO.md (sembra non rilevante)
- ❌ REORGANIZATION_PLAN.md (questo file, temporaneo)
- ❌ temp_*.jsonl (file temporanei)
- ❌ batch_*.jsonl (file temporanei)

---

## 📝 Aggiornamenti Necessari Dopo lo Spostamento

### 1. README.md
```markdown
# Links da aggiornare:
- [LLM_QUICKSTART.md](LLM_QUICKSTART.md) → [docs/LLM_QUICKSTART.md](docs/LLM_QUICKSTART.md)
- [docs/recommended_models.md](docs/recommended_models.md) → OK
```

### 2. Makefile
```makefile
# Aggiornare path:
llm-server:
    python examples/llm_server_example.py  # invece di ./llm_server_example.py

analyze:
    python scripts/analyze_extraction.py   # invece di ./analyze_extraction.py
```

### 3. Documentazione Interna
Aggiornare link relativi in:
- docs/LLM_QUICKSTART.md
- docs/SETUP_GPT4MINI.md
- docs/IMPLEMENTATION_SUMMARY.md
- docs/FINAL_SUMMARY.md

### 4. .gitignore
```
# Aggiungi:
.env
.env.*
*.env

outputs/*.jsonl
!outputs/.gitkeep

.llm_cache/
temp_*.jsonl
batch_*.jsonl
```

---

## ✅ Checklist Post-Riorganizzazione

- [ ] Tutti i file spostati correttamente
- [ ] Link nei README aggiornati
- [ ] Makefile aggiornato
- [ ] .gitignore aggiornato
- [ ] Test che i comandi funzionino ancora
- [ ] Documentazione coerente

---

## 🚀 Esecuzione

Vedi: `execute_reorganization.sh`
