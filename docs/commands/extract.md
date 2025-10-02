# robimb extract - Estrazione Proprietà

Estrae proprietà da descrizioni BIM utilizzando una pipeline configurabile con regole, QA encoder e LLM.

## Sintassi

```bash
robimb extract properties [OPTIONS]
```

## Opzioni Principali

### Input/Output
- `--input PATH`: File JSONL di input con descrizioni BIM
- `--output PATH`: File JSONL di output con proprietà estratte
- `--sample N`: Processa solo i primi N record (utile per test)

### Pipeline di Estrazione

#### Regole (sempre attive)
Matching basato su pattern e lessici predefiniti.

#### QA Encoder (opzionale)
- `--use-qa`: Abilita QA encoder
- `--qa-model-dir PATH`: Directory del modello QA fine-tuned
- `--no-qa`: Disabilita QA encoder (default)

#### LLM (opzionale)
- `--llm-endpoint URL`: Endpoint del servizio LLM
- `--llm-model NAME`: Nome del modello LLM da usare
- `--llm-timeout SECONDS`: Timeout richieste LLM (default: 30)

### Configurazione Avanzata
- `--registry PATH`: Path al registry delle proprietà (default: da config)
- `--null-threshold FLOAT`: Soglia per considerare una risposta come NULL (default: 0.25)

## Esempi

### Solo regole (più veloce, baseline)
```bash
robimb extract properties \
  --input data/descriptions.jsonl \
  --output outputs/rules_only.jsonl \
  --no-qa
```

### Regole + LLM
```bash
robimb extract properties \
  --input data/descriptions.jsonl \
  --output outputs/with_llm.jsonl \
  --llm-endpoint http://localhost:8000/extract \
  --llm-model gpt-4o-mini \
  --no-qa \
  --sample 100
```

### Pipeline completa (regole + QA + LLM)
```bash
robimb extract properties \
  --input data/descriptions.jsonl \
  --output outputs/full_pipeline.jsonl \
  --use-qa \
  --qa-model-dir outputs/qa_models/base \
  --llm-endpoint http://localhost:8000/extract \
  --llm-model gpt-4o-mini
```

## Formato Output

Ogni riga del file JSONL di output contiene:

```json
{
  "description": "Testo originale",
  "category": "categoria_bim",
  "properties": {
    "dimensioni": {"value": "120x60", "confidence": 0.95},
    "materiale": {"value": "gres porcellanato", "confidence": 0.88}
  },
  "extraction_metadata": {
    "rules_count": 2,
    "qa_count": 1,
    "llm_count": 0,
    "total_time_ms": 450
  }
}
```

## Workflow con Makefile

Il Makefile fornisce shortcut utili:

```bash
# Solo regole
make extract-rules SAMPLE=50

# Con LLM
make extract-llm SAMPLE=50

# Pipeline completa
make extract-full SAMPLE=50

# Confronto rules vs LLM
make compare-llm SAMPLE=100
```

## Performance e Tuning

- **Solo regole**: ~100-200 doc/sec
- **Con QA encoder**: ~20-50 doc/sec (dipende da GPU)
- **Con LLM**: ~1-5 doc/sec (dipende da endpoint)

Usa `--sample` per test rapidi prima di processare l'intero dataset.

## Analisi Risultati

```bash
python scripts/analysis/extraction_results.py outputs/extracted.jsonl
```

## Troubleshooting

### LLM endpoint non risponde
```bash
# Verifica connettività
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"prompt":"test","schema":{},"model":"test"}'
```

### QA model non trovato
```bash
# Verifica path
robimb config inspect
```

## Vedi Anche

- [Guida LLM Integration](../guides/llm_integration.md)
- [Production Setup](../guides/production_resource_setup.md)
