# roBERT - BIM NLP Toolkit

Toolkit Python industriale per l'estrazione di proprietà e la classificazione di descrizioni BIM, con supporto per TAPT (Task-Adaptive Pre-Training) e classificatori gerarchici.

## 🚀 Quick Start

### Installazione

```bash
# Clona repository
git clone https://github.com/atipiqal/roBERT.git
cd roBERT

# Installa dipendenze
pip install -e ".[dev]"

# Verifica installazione
robimb --help
```

### Primo Utilizzo

```bash
# Estrazione proprietà (solo regole)
robimb extract properties \
  --input data/descriptions.jsonl \
  --output outputs/extracted.jsonl \
  --sample 10

# Analisi risultati
python scripts/analysis/extraction_results.py outputs/extracted.jsonl
```

## 📂 Struttura Progetto

```
roBERT/
├── src/robimb/              # Codice sorgente principale
│   ├── cli/                 # Command-line interface
│   ├── extraction/          # Pipeline estrazione proprietà
│   ├── inference/           # Moduli di inferenza
│   ├── models/              # Definizioni modelli ML
│   ├── training/            # Training e fine-tuning
│   ├── registry/            # Gestione risorse e schemi
│   ├── reporting/           # Generazione report
│   └── utils/               # Utilities
│
├── resources/               # Risorse produzione
│   ├── config/              # Configurazioni
│   └── data/properties/     # Lexicon, schemi, registry, prompts
│
├── scripts/                 # Script di supporto
│   ├── analysis/            # Analisi dataset e risultati
│   ├── testing/             # Testing e validazione
│   └── setup/               # Setup e configurazione
│
├── docs/                    # Documentazione
│   ├── commands/            # Documentazione CLI
│   ├── guides/              # Guide tecniche
│   └── development/         # Guide sviluppo
│
├── tests/                   # Test suite
├── outputs/                 # Output generati (gitignored)
└── Makefile                 # Automazione task comuni
```

## 📚 Documentazione

### Comandi CLI

| Comando | Descrizione | Docs |
|---------|-------------|------|
| `robimb extract` | Estrazione proprietà da descrizioni BIM | [extract.md](docs/commands/extract.md) |
| `robimb convert` | Conversione dataset e label map | [convert.md](docs/commands/convert.md) |
| `robimb train` | Training modelli (label/gerarchico/TAPT) | [train.md](docs/commands/train.md) |
| `robimb evaluate` | Valutazione performance modelli | [evaluate.md](docs/commands/evaluate.md) |
| `robimb pack` | Creazione knowledge pack | [pack.md](docs/commands/pack.md) |
| `robimb config` | Ispezione configurazione | [config.md](docs/commands/config.md) |

**Panoramica completa**: [docs/commands/overview.md](docs/commands/overview.md)

### Guide Tecniche

- [Quick Start GPT-4o-mini](QUICKSTART_GPT4MINI.md) - ⚡ Setup GPT-4o-mini in 5 minuti
- [Architettura del Sistema](ARCHITECTURE.md) - Overview tecnico e design
- [Orchestration Improvements](docs/guides/ORCHESTRATION_IMPLEMENTATION.md) - ⭐ Domain heuristics per estrazione migliorata
- [Production Setup](docs/guides/production_resource_setup.md) - Setup ambiente produzione
- [LLM Integration](examples/README.md) - Server LLM (mock e GPT-4o-mini)
- [Scripts README](scripts/README.md) - Documentazione script di supporto

## 🔧 Workflow Tipico

### 1. Preparazione Dataset
```bash
robimb convert \
  --input data/raw/dataset.jsonl \
  --output data/processed/dataset.jsonl \
  --label-map data/processed/label_map.json
```

### 2. Training Modello
```bash
robimb train label \
  --train-data data/processed/train.jsonl \
  --eval-data data/processed/eval.jsonl \
  --model-name atipiqal/BOB \
  --output-dir outputs/models/label
```

### 3. Creazione Knowledge Pack
```bash
robimb pack \
  --resources-dir resources/data/properties \
  --output outputs/knowledge_pack.tar.gz
```

### 4. Estrazione Proprietà
```bash
# Solo regole (veloce, baseline)
robimb extract properties \
  --input data/descriptions.jsonl \
  --output outputs/extracted.jsonl \
  --no-qa

# Con LLM (migliore qualità)
robimb extract properties \
  --input data/descriptions.jsonl \
  --output outputs/extracted.jsonl \
  --llm-endpoint http://localhost:8000/extract \
  --llm-model gpt-4o-mini \
  --no-qa

# Pipeline completa (regole + QA + LLM)
robimb extract properties \
  --input data/descriptions.jsonl \
  --output outputs/extracted.jsonl \
  --use-qa \
  --qa-model-dir outputs/models/label \
  --llm-endpoint http://localhost:8000/extract \
  --llm-model gpt-4o-mini
```

### 5. Valutazione
```bash
robimb evaluate \
  --model-dir outputs/models/label \
  --test-data data/processed/test.jsonl \
  --output outputs/metrics.json
```

## 🛠️ Makefile - Task Comuni

Il Makefile fornisce shortcut per operazioni frequenti:

```bash
# Sviluppo
make install              # Installa dipendenze
make tests                # Esegui test suite
make clean                # Pulizia file temporanei

# LLM Integration
make llm-server           # Avvia mock LLM server
make llm-test             # Testa connessione LLM
make llm-integration-test # Test completo integrazione

# Estrazione
make extract-rules SAMPLE=50      # Estrazione solo regole
make extract-llm SAMPLE=50        # Estrazione con LLM
make extract-full SAMPLE=50       # Pipeline completa
make compare-llm SAMPLE=100       # Confronto rules vs LLM

# Analisi
make analyze OUTPUT=outputs/extracted.jsonl

# Help
make help                 # Mostra tutti i comandi disponibili
```

## 🔬 Scripts di Analisi

```bash
# Analisi dataset
python scripts/analysis/dataset_basic.py data/dataset.jsonl
python scripts/analysis/dataset_detailed.py data/dataset.jsonl

# Analisi risultati estrazione
python scripts/analysis/extraction_results.py outputs/extracted.jsonl

# Testing
python scripts/testing/sanity_check.py
bash scripts/testing/test_llm_integration.sh
```

## ⚙️ Configurazione

### Variabili d'Ambiente

```bash
export ROBIMB_REGISTRY_PATH=resources/data/properties/registry.json
export ROBIMB_QA_MODEL_DIR=outputs/qa_models/base
export ROBIMB_NULL_THRESHOLD=0.25
```

### File di Configurazione

Esempio `resources/config/production.toml`:

```toml
[paths]
registry = "resources/data/properties/registry.json"
qa_model = "outputs/qa_models/base"

[extraction]
null_threshold = 0.25
llm_timeout = 30

[llm]
endpoint = "http://localhost:8000/extract"
model = "gpt-4o-mini"
```

Usa con:
```bash
robimb --config resources/config/production.toml extract properties ...
```

## 📦 Risorse Principali

### Lexicon
- **Brands**: [resources/data/properties/lexicon/brands.json](resources/data/properties/lexicon/brands.json)
- **Materials**: [resources/data/properties/lexicon/materials.json](resources/data/properties/lexicon/materials.json)
- **Norms**: [resources/data/properties/lexicon/norms.json](resources/data/properties/lexicon/norms.json)
- **Colors (RAL)**: [resources/data/properties/lexicon/colors_ral.json](resources/data/properties/lexicon/colors_ral.json)

### Schema
Definizioni proprietà per categoria BIM in [resources/data/properties/schema/](resources/data/properties/schema/)

### Registry
Mapping proprietà-categoria: [resources/data/properties/registry.json](resources/data/properties/registry.json)

## 🧪 Testing

```bash
# Test suite completa
pytest

# Test specifici
pytest tests/test_extraction.py
pytest tests/test_models.py

# Con coverage
pytest --cov=robimb --cov-report=html
```

## 📊 Performance

- **Solo regole**: ~100-200 doc/sec
- **Con QA encoder**: ~20-50 doc/sec (GPU-dependent)
- **Con LLM**: ~1-5 doc/sec (endpoint-dependent)

## 🤝 Contribuire

1. Fork del repository
2. Crea feature branch (`git checkout -b feature/nome-feature`)
3. Commit modifiche (`git commit -m 'Add: nuova feature'`)
4. Push al branch (`git push origin feature/nome-feature`)
5. Apri Pull Request

## 📝 Licenza

Proprietary - Copyright (c) atipiqal

## 🔗 Risorse Correlate

- **Repository**: https://github.com/atipiqal/roBERT
- **Issues**: https://github.com/atipiqal/roBERT/issues
- **Documentazione**: [docs/](docs/)
