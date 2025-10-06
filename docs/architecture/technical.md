# Architettura roBERT

Panoramica tecnica del sistema roBERT per l'estrazione di proprietà e classificazione di prodotti edili.

## 🎯 Overview

roBERT è una toolkit modulare che combina:
- **Regole deterministiche** (parsers basati su pattern)
- **ML encoder** (QA models fine-tuned)
- **LLM** (integrazione opzionale via API)

per estrarre proprietà strutturate da descrizioni testuali non strutturate di prodotti edili.

## 🏗️ Architettura a Livelli

```
┌─────────────────────────────────────────────────────┐
│                  CLI Layer                          │
│              (robimb.cli.*)                         │
│  extract | convert | train | evaluate | pack       │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│              Orchestration Layer                    │
│          (robimb.extraction.orchestrator)           │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │   Rules     │→ │  QA Encoder  │→ │    LLM    │ │
│  │  (parsers)  │  │ (property_qa)│  │  (qa_llm) │ │
│  └─────────────┘  └──────────────┘  └───────────┘ │
│                                                     │
│  Schema Registry → Fusion → Validation              │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│               Core Components                       │
│                                                     │
│  • Parsers (dimensions, units, colors, ...)        │
│  • Matchers (brands, materials, norms)             │
│  • Models (label_model, masked_model)              │
│  • Training (label_trainer, hier_trainer, tapt)    │
│  • Registry (schema, validators, loaders)          │
│  • Utils (metrics, logging, data prep)             │
└─────────────────────────────────────────────────────┘
```

## 📦 Moduli Principali

### 1. CLI Layer (`src/robimb/cli/`)

Entry point per tutte le operazioni utente.

| Modulo | Responsabilità |
|--------|----------------|
| `main.py` | CLI principale, routing comandi |
| `extract.py` | Estrazione proprietà da descrizioni |
| `convert.py` | Conversione dataset e label map |
| `train.py` | Training modelli ML |
| `evaluate.py` | Valutazione performance |
| `pack.py` | Packaging risorse per produzione |
| `config.py` | Ispezione configurazione |

### 2. Extraction Layer (`src/robimb/extraction/`)

Pipeline di estrazione ibrida con fusione multi-sorgente.

#### Core Files

| File | Responsabilità |
|------|----------------|
| `orchestrator.py` | Coordinamento pipeline completa |
| `orchestrator_base.py` | Logica base orchestrazione |
| `orchestrator_async.py` | Variante asincrona (LLM batch) |
| `fuse.py` | Fusione risultati da multiple sorgenti |
| `fusion_policy.py` | Policy per conflitti e priorità |
| `schema_registry.py` | Gestione schemi proprietà per categoria |
| `property_qa.py` | QA encoder (BERT fine-tuned) |
| `qa_llm.py` | Client per endpoint LLM esterni |
| `prompts.py` | Template prompt per LLM |
| `validators.py` | Validazione output estratti |
| `normalize.py` | Normalizzazione valori |
| `lexicon.py` | Caricamento lessici (norms, producers) |
| `legacy.py` | Engine regex legacy (deprecated) |

#### Parsers (`parsers/`)

Parser deterministici basati su pattern regex e logica custom.

| Parser | Estrae |
|--------|--------|
| `dimensions.py` | Dimensioni (LxWxH, diametri, etc.) |
| `units.py` | Unità di misura e conversioni |
| `numbers.py` | Numeri con locale italiano |
| `labeled_dimensions.py` | Dimensioni con label (spessore=12mm) |
| `thickness.py` | Spessori specifici |
| `colors.py` | Codici colore RAL |
| `standards.py` | Standard e norme (ISO, EN, UNI) |
| `fire_class.py` | Classi di resistenza al fuoco |
| `thermal.py` | Proprietà termiche (U, R, λ) |
| `acoustic.py` | Proprietà acustiche (Rw, ΔLw) |
| `sound_insulation.py` | Isolamento acustico |
| `installation_type.py` | Tipo installazione |
| `flow_rate.py` | Portate (l/min, m³/h) |

#### Matchers (`matchers/`)

Matching basato su dizionari/lexicon.

| Matcher | Estrae |
|---------|--------|
| `brands.py` | Brand e produttori |
| `materials.py` | Materiali (fuzzy matching) |
| `norms.py` | Norme tecniche |

### 3. Models Layer (`src/robimb/models/`)

Definizioni modelli PyTorch per classificazione ed estrazione.

| Modulo | Descrizione |
|--------|-------------|
| `label_model.py` | Classificatore flat (multi-label) |
| `masked_model.py` | Classificatore gerarchico con maschere ontologiche |
| `span_extractor.py` | QA-based model per estrazione span di proprietà |
| `price_regressor.py` | Modello regressione prezzi unit-aware |

### 4. Training Layer (`src/robimb/training/`)

Training modelli per span extraction e price prediction.

| Modulo | Descrizione |
|--------|-------------|
| `span_trainer.py` | Training span extractor (QA-based) |
| `price_trainer.py` | Training price regressor (unit-aware) |
| `property_utils.py` | Utilities per property extraction training |

### 5. Inference Layer (`src/robimb/inference/`)

Inferenza con modelli addestrati.

| Modulo | Descrizione |
|--------|-------------|
| `span_inference.py` | Inferenza span extraction |
| `price_inference.py` | Inferenza price prediction |
| `category.py` | Predizione categorie (LabelEmbed/HF) |
| `calibration.py` | Calibrazione probabilità |

### 6. Registry Layer (`src/robimb/registry/`)

Gestione risorse e configurazioni.

| Modulo | Descrizione |
|--------|-------------|
| `schemas.py` | Schema Pydantic per risorse |
| `loader.py` | Caricamento risorse (registry, lexicon) |
| `validators.py` | Validazione risorse |
| `normalizers.py` | Normalizzatori per diverse tipologie |

### 7. Utils Layer (`src/robimb/utils/`)

Utilities trasversali.

| Modulo | Descrizione |
|--------|-------------|
| `dataset_prep.py` | Preparazione dataset per training |
| `metrics_utils.py` | Calcolo metriche (F1, accuracy, etc.) |
| `ontology_utils.py` | Gestione gerarchie ontologiche |
| `registry_io.py` | I/O registry e risorse |
| `packing_utils.py` | Utilities per knowledge packing |
| `logging.py` | Logging configurabile |
| `data_utils.py` | Utilities manipolazione dati |
| `io_utils.py` | I/O generico |
| `sampling.py` | Sampling dataset |

## 🔄 Flusso di Estrazione

### Pipeline Completa (Rules → QA → LLM)

```
Input: "Piastrella in gres porcellanato 60x60 cm sp. 10mm classe PEI 4"
   │
   ├─→ [1] Schema Registry
   │      └─→ Carica schema per categoria "opere_di_pavimentazione"
   │
   ├─→ [2] Rules Extractor (parsers + matchers)
   │      ├─→ DimensionParser    → {dimensioni: "60x60"}
   │      ├─→ ThicknessParser    → {spessore: "10mm"}
   │      ├─→ MaterialMatcher    → {materiale: "gres porcellanato"}
   │      └─→ StandardsParser    → {classe_pei: "4"}
   │
   ├─→ [3] QA Encoder (opzionale, se --use-qa)
   │      └─→ BERT fine-tuned interroga testo per proprietà mancanti
   │
   ├─→ [4] LLM (opzionale, se --llm-endpoint)
   │      └─→ GPT-4o-mini estrae proprietà non coperte da rules/QA
   │
   ├─→ [5] Fusion
   │      └─→ Combina risultati con policy (rules > QA > LLM)
   │
   ├─→ [6] Validation
   │      └─→ Verifica conformità a schema e tipi
   │
   └─→ Output: {
         "dimensioni": {"value": "60x60", "source": "rules", "confidence": 1.0},
         "spessore": {"value": "10", "unit": "mm", "source": "rules", "confidence": 0.95},
         "materiale": {"value": "gres porcellanato", "source": "rules", "confidence": 0.88},
         "classe_pei": {"value": "4", "source": "rules", "confidence": 0.92}
       }
```

### Fusion Policy

La fusione risolve conflitti secondo priorità:

1. **Rules** (massima confidence, deterministico)
2. **QA Encoder** (ML-based, supervised)
3. **LLM** (fallback per casi complessi)

Se multiple sorgenti estraggono la stessa proprietà, vince quella con priorità maggiore.

## 🗄️ Risorse

### Registry Structure

```json
{
  "controsoffitti": {
    "properties": [
      {
        "name": "dimensioni",
        "type": "string",
        "extractors": ["dimensions_parser"],
        "required": false,
        "prompt_template": "extract_dimensions"
      },
      ...
    ]
  }
}
```

### Lexicon Structure

```
resources/data/properties/lexicon/
├── brands.json              # {"Mapei": {...}, "Kerakoll": {...}}
├── materials.json           # {"gres porcellanato": {...}, ...}
├── norms.json               # {"UNI EN 14411": {...}, ...}
├── colors_ral.json          # {"RAL 9016": "bianco", ...}
├── producers_by_category.json
├── norms_by_category.json
└── standards_prefixes.json  # {"ISO": {...}, "EN": {...}}
```

### Schema Structure

```
resources/data/properties/schema/
├── controsoffitti.json
├── opere_di_pavimentazione.json
├── opere_di_rivestimento.json
├── apparecchi_sanitari_accessori.json
└── ...
```

Ogni schema definisce:
- Lista proprietà per categoria
- Tipo dato (string, number, boolean, enum)
- Extractors applicabili
- Prompt template per LLM
- Vincoli di validazione

## 🔌 Estensibilità

### Aggiungere un Nuovo Parser

1. Crea `src/robimb/extraction/parsers/new_parser.py`:

```python
from typing import Optional
import re

def parse_new_property(text: str) -> Optional[dict]:
    """Extract new property from text."""
    pattern = r"pattern_here"
    match = re.search(pattern, text)
    if match:
        return {
            "value": match.group(1),
            "confidence": 0.9,
            "source": "rules"
        }
    return None
```

2. Registra in `extractors.json`:

```json
{
  "new_property_parser": {
    "module": "robimb.extraction.parsers.new_parser",
    "function": "parse_new_property"
  }
}
```

3. Aggiungi a schema categoria:

```json
{
  "properties": [
    {
      "name": "new_property",
      "type": "string",
      "extractors": ["new_property_parser"]
    }
  ]
}
```

### Aggiungere una Nuova Categoria

1. Crea schema: `resources/data/properties/schema/nuova_categoria.json`
2. Registra in `registry.json`
3. Opzionale: aggiungi lexicon specifici

## 📊 Performance Considerations

### Bottleneck Analysis

| Component | Throughput | Latency | Bottleneck |
|-----------|------------|---------|------------|
| Rules | ~200 doc/sec | <5ms | Regex compilation |
| QA Encoder | ~50 doc/sec | ~20ms | GPU inference |
| LLM | ~3 doc/sec | ~300ms | API network |

### Ottimizzazioni

- **Batching**: QA encoder e LLM supportano batching
- **Caching**: LLM responses cached per descrizioni identiche
- **Async**: `orchestrator_async.py` per parallelizzazione LLM
- **Sampling**: Usa `--sample N` per test rapidi

## 🧪 Testing Strategy

```
tests/
├── test_parsers.py        # Unit test parser individuali
├── test_matchers.py       # Unit test matchers
├── test_orchestrator.py   # Integration test pipeline
├── test_models.py         # Test modelli ML
├── test_registry.py       # Test caricamento risorse
└── test_cli.py            # Test comandi CLI
```

Esegui con:
```bash
pytest                          # Tutto
pytest tests/test_parsers.py    # Specifico
pytest --cov=robimb            # Con coverage
```

## 🔐 Security & Production

### Best Practices

1. **Validazione Input**: Tutti gli input passano attraverso Pydantic validators
2. **Timeout LLM**: Configurabile con `--llm-timeout`
3. **Rate Limiting**: Implementato nel client LLM
4. **Error Handling**: Graceful degradation (LLM fail → fallback a QA/rules)
5. **Logging**: Structured logging per audit trail

### Deployment

Vedi [docs/guides/production_resource_setup.md](docs/guides/production_resource_setup.md)

## 📚 Riferimenti

- [README.md](README.md): Quick start e overview
- [docs/commands/](docs/commands/): Documentazione CLI
- [scripts/README.md](scripts/README.md): Script di supporto
- [Makefile](Makefile): Automazione task comuni
