# roBERT

Sistema di classificazione ed estrazione di proprietà tecniche da descrizioni di materiali edilizi.

## Struttura del progetto

```
roBERT/
├── src/robimb/              # Codice sorgente principale
│   ├── cli/                 # Command-line interface
│   ├── models/              # Modelli BERT (label_model, masked_model)
│   ├── training/            # Training pipeline (hier_trainer, tapt_mlm)
│   ├── inference/           # Inferenza e calibrazione
│   ├── extraction/          # Estrazione proprietà (parsers, matchers, QA-LLM)
│   │   ├── parsers/         # Parser specializzati (units, colors, standards, etc.)
│   │   └── matchers/        # Matcher per materiali, brand, norme
│   ├── registry/            # Registry delle proprietà e validatori
│   ├── reporting/           # Generazione report dataset/predizioni
│   └── utils/               # Utility (data, metrics, ontology, logging)
│
├── data/                    # Dataset (raw, processed, train/test splits)
├── outputs/                 # Output di training e inferenza
├── pack/                    # Knowledge pack (registry e extractors JSON)
│   └── current/             # Pack corrente in uso
│
├── scripts/                 # Script di utilità
│   ├── analysis/            # Script di analisi dataset
│   └── sanity_check.py      # Controlli di validità
│
├── docs/                    # Documentazione
│   ├── reports/             # Report di analisi
│   ├── ADR/                 # Architecture Decision Records
│   └── *.md                 # Guide CLI, overview, roadmap
│
├── tests/                   # Test suite
├── pyproject.toml           # Configurazione progetto e dipendenze
└── README.md                # Questo file
```

## Componenti principali

### CLI
- **train**: Training del modello di classificazione
- **extract**: Estrazione proprietà da testo
- **convert**: Conversione formato dataset
- **evaluate**: Valutazione modello
- **pack**: Gestione knowledge pack

### Extraction Pipeline
Sistema multi-stage per estrazione proprietà:
1. **Parser**: Pattern regex per valori numerici, colori, unità di misura
2. **Matcher**: Ricerca lessicale per marchi, materiali, norme tecniche
3. **QA-LLM**: Estrazione via LLM per campi complessi
4. **Fusion**: Aggregazione e normalizzazione risultati

### Registry
Schema delle proprietà per categoria con:
- Definizione tipi e vincoli
- Enum per valori ammessi
- Normalizzatori e validatori
- Configurazione sorgenti estrazione

## Installazione

```bash
pip install -e .
```

## Uso rapido

```bash
# Training modello classificazione
robimb train --config configs/train_config.yaml

# Estrazione proprietà
robimb extract --input data/raw/descriptions.jsonl --output results.jsonl

# Valutazione modello
robimb evaluate --test-data data/test.jsonl --model-path outputs/model/
```

## Documentazione

Vedi [docs/index.md](docs/index.md) per la documentazione completa.

## Knowledge Pack

La cartella `pack/current/` contiene i file JSON con:
- **registry.json**: Schema proprietà per categoria
- **extractors.json**: Pattern regex e normalizzatori per parser

Questi file configurano il comportamento dell'extraction pipeline.

## Licenza

[Da specificare]
