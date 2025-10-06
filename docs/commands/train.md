# robimb train - Training Modelli

Training di modelli per estrazione proprietà e predizione prezzi.

## Sintassi

```bash
robimb train [SUBCOMMAND] [OPTIONS]
```

## Subcomandi

### Span Extraction Training
Training del modello QA-based per estrazione span di proprietà.

```bash
robimb train span \
  --train-data data/qa_data.jsonl \
  --output-dir outputs/span_model \
  --backbone-name atipiqal/BOB \
  --epochs 3 \
  --batch-size 8
```

### Price Regression Training
Training del modello per predizione prezzi unit-aware.

```bash
robimb train price \
  --train-data data/price_data.jsonl \
  --output-dir outputs/price_model \
  --use-properties \
  --epochs 10 \
  --batch-size 16
```

## Opzioni Comuni

- `--train-data PATH`: Dataset di training
- `--eval-data PATH`: Dataset di validazione
- `--model-name NAME`: Modello base (HuggingFace)
- `--output-dir PATH`: Directory per salvare il modello
- `--batch-size INT`: Batch size (default: 16)
- `--epochs INT`: Numero di epoche (default: 3)
- `--learning-rate FLOAT`: Learning rate (default: 2e-5)

## Preparazione Dataset

Prima del training, usa `robimb prepare` per preparare i dati:

```bash
# Prepara dataset per span extraction
robimb prepare span \
  --input data/raw.jsonl \
  --output data/qa_data.jsonl

# Prepara dataset per price regression
robimb prepare price \
  --input data/raw.csv \
  --output data/price_data.jsonl

# Prepara tutti i dataset
robimb prepare all \
  --input data/raw.jsonl \
  --output-dir data/prepared/
```

## Vedi Anche

- [convert.md](convert.md): Preparazione dataset
- [evaluate.md](evaluate.md): Valutazione modelli
