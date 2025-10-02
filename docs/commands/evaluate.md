# robimb evaluate - Valutazione Modelli

Valuta performance di modelli su dataset di test.

## Sintassi

```bash
robimb evaluate [OPTIONS]
```

## Opzioni

- `--model-dir PATH`: Directory del modello da valutare
- `--test-data PATH`: Dataset di test
- `--output PATH`: File di output con metriche
- `--metrics LIST`: Metriche da calcolare (accuracy,f1,precision,recall)

## Esempio

```bash
robimb evaluate \
  --model-dir outputs/models/label \
  --test-data data/test.jsonl \
  --output outputs/metrics.json \
  --metrics accuracy,f1,precision,recall
```

## Output

Il comando genera un report JSON con:

```json
{
  "accuracy": 0.92,
  "f1_macro": 0.88,
  "f1_micro": 0.91,
  "precision": 0.89,
  "recall": 0.87,
  "per_class_metrics": {...}
}
```

## Vedi Anche

- [train.md](train.md): Training modelli
