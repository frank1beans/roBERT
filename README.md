# roBERT - NLP Toolkit per l'Edilizia

Toolkit Python industriale per l'estrazione di proprietà e la classificazione di descrizioni di prodotti edili, con supporto per TAPT (Task-Adaptive Pre-Training) e classificatori gerarchici.

## ⭐ Features Principali

### 🎯 Modelli ML Disponibili

#### 1. **Classificazione**
- **Label Embedding**: Classificatore multi-label gerarchico
- **Hierarchical Model**: Modello con masked training gerarchico

#### 2. **Span Extraction**
**Estrazione intelligente context-aware** basata su QA-style deep learning
- ✅ **Alta precisione**: ~90% accuracy con confidence scores
- ✅ **Context-aware**: Distingue contesto prodotto vs descrizione generica
- ✅ **Zero falsi positivi**: Previene estrazioni spurie
- 🔧 Training: `robimb train span --train-data qa_data.jsonl --output-dir outputs/span_model`
- 🚀 Inferenza: `robimb extract predict-spans --model-dir outputs/span_model --input data.jsonl`

#### 3. **Price Regression**
**Predizione prezzi unit-aware** basata su descrizione + proprietà estratte
- 💰 **Unit-aware**: Distingue scale diverse (mm vs m, kg vs g)
- 📊 **Property conditioning**: Usa proprietà estratte per migliore accuratezza
- 📈 **Metriche**: MAPE, RMSE, MAE su scala log
- 🔧 Training: `robimb train price --train-data price_data.jsonl --output-dir outputs/price_model`
- 🚀 Inferenza: `robimb predict price --model-dir outputs/price_model --input data.jsonl`

### 🔄 Pipeline End-to-End

```python
# 1. Classificazione categoria
from robimb.models.label_model import load_label_embed_model
classifier = load_label_embed_model("atipiqal/RoBERTino")

# 2. Span extraction
from robimb.inference.span_inference import SpanInference
span_extractor = SpanInference("outputs/span_model")

# 3. Price prediction
from robimb.inference.price_inference import PriceInference
price_predictor = PriceInference("outputs/price_model")

# Pipeline completa
text = "Pavimento gres Florim 120x280 cm, spessore 6mm"
category = classifier.predict(text)
properties = span_extractor.extract_properties(text)
price = price_predictor.predict(text, properties)
```

**📚 [Centro Documentazione](docs/README.md)** - Guide complete per tutti i livelli

---

## 🚀 Quick Start

**Nuovo a roBERT?** → Inizia dalla [**Guida Introduttiva**](docs/getting-started/README.md) pensata per tutti (anche non tecnici)

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

**[📖 Centro Documentazione Completo](docs/README.md)** - Hub centrale organizzato per tipo di utente

### 🎯 Guide per Iniziare (Per Tutti)

<table>
<tr>
<td width="33%">

**[🚀 Introduzione](docs/getting-started/README.md)**

Cos'è roBERT, come funziona, esempi pratici
*(Non richiede conoscenze tecniche)*

</td>
<td width="33%">

**[📦 Installazione](docs/getting-started/installation.md)**

Setup guidato passo-passo
*(Windows, macOS, Linux)*

</td>
<td width="33%">

**[🔄 Workflow](docs/getting-started/workflows.md)**

Casi d'uso comuni, comandi pratici
*(Con esempi reali)*

</td>
</tr>
</table>

### 📖 Risorse per Ruolo

| Ruolo | Documenti Consigliati |
|-------|----------------------|
| 👤 **Business User** | [Intro](docs/getting-started/README.md) → [Workflow](docs/getting-started/workflows.md) → [Extract](docs/commands/extract.md) |
| 💻 **Sviluppatore** | [Architettura](docs/architecture/technical.md) → [Pipeline](docs/architecture/pipeline.md) → [Comandi CLI](docs/commands/overview.md) |
| 🧑‍🔬 **ML Engineer** | [Training Roadmap](docs/models/training-roadmap.md) → [Span Extractor](docs/models/span-extraction.md) → [Price Regressor](docs/models/price-regression.md) |
| 🏢 **DevOps** | [Installazione](docs/getting-started/installation.md) → [Production Setup](docs/guides/production_resource_setup.md) → [Config](docs/commands/config.md) |

### ⚙️ Riferimenti Rapidi

**Comandi:** [extract](docs/commands/extract.md) | [predict](docs/commands/predict.md) | [train](docs/commands/train.md) | [tutti](docs/commands/overview.md)

**Architettura:** [Overview](docs/architecture/overview.md) | [Tecnica](docs/architecture/technical.md) | [Pipeline](docs/architecture/pipeline.md)

**Modelli:** [Span Extractor](docs/models/span-extraction.md) | [Price Regressor](docs/models/price-regression.md) | [Training](docs/models/training-roadmap.md)

## 🔧 Workflow Tipico

### 1. Preparazione Dataset
```bash
robimb convert \
  --input data/raw/dataset.jsonl \
  --output data/processed/dataset.jsonl \
  --label-map data/processed/label_map.json
```

### 2. Training Modelli

```bash
# Classificatore label embedding
robimb train label \
  --train-jsonl data/train.jsonl \
  --val-jsonl data/val.jsonl \
  --base-model atipiqal/BOB \
  --out-dir outputs/label_model

# Span extractor
robimb train span \
  --train-data data/qa_dataset.jsonl \
  --output-dir outputs/span_model \
  --backbone-name atipiqal/BOB

# Price regressor
robimb train price \
  --train-data data/price_data.jsonl \
  --output-dir outputs/price_model \
  --use-properties
```

### 3. Creazione Knowledge Pack
```bash
robimb pack \
  --resources-dir resources/data/properties \
  --output outputs/knowledge_pack.tar.gz
```

### 4. Estrazione Proprietà

```bash
# Estrazione base (rules + matchers)
robimb extract properties \
  --input data/descriptions.jsonl \
  --output outputs/extracted.jsonl

# Span-based extraction
robimb extract predict-spans \
  --model-dir outputs/span_model \
  --input data/descriptions.jsonl \
  --output outputs/spans.jsonl \
  --properties marchio,materiale,dimensione_lunghezza

# Price prediction
robimb predict price \
  --model-dir outputs/price_model \
  --input outputs/spans.jsonl \
  --output outputs/with_prices.jsonl

# Con LLM (opzionale, migliore qualità)
robimb extract properties \
  --input data/descriptions.jsonl \
  --output outputs/extracted_llm.jsonl \
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
Definizioni proprietà per categoria in [resources/data/properties/schema/](resources/data/properties/schema/)

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
