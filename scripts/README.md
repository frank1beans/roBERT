# Scripts

Script di supporto per analisi, testing e setup del progetto roBERT.

## Struttura

```
scripts/
├── analysis/           # Analisi dataset e risultati
├── testing/            # Script per testing e validazione
└── setup/              # Script di setup e configurazione
```

## Analysis

### dataset_basic.py
Analisi base di dataset BIM.

```bash
python scripts/analysis/dataset_basic.py data/dataset.jsonl
```

### dataset_detailed.py
Analisi dettagliata con statistiche avanzate.

```bash
python scripts/analysis/dataset_detailed.py data/dataset.jsonl
```

### extraction_results.py
Analisi risultati di estrazione proprietà.

```bash
python scripts/analysis/extraction_results.py outputs/extracted.jsonl
```

## Testing

### sanity_check.py
Verifica integrità configurazione e risorse.

```bash
python scripts/testing/sanity_check.py
```

### test_llm_integration.sh
Test completo integrazione LLM endpoint.

```bash
bash scripts/testing/test_llm_integration.sh
```

## Setup

### setup_gpt4mini.ps1
Setup ambiente per GPT-4o-mini (Windows PowerShell).

```powershell
.\scripts\setup\setup_gpt4mini.ps1
```

## Vedi Anche

- [Makefile](../Makefile): automazione task comuni
- [docs/commands/](../docs/commands/): documentazione CLI
