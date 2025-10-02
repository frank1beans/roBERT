# robimb convert - Conversione Dataset

Converte dataset per training e genera label map.

## Sintassi

```bash
robimb convert [OPTIONS]
```

## Opzioni

- `--input PATH`: Dataset di input
- `--output PATH`: Dataset di output convertito
- `--format FORMAT`: Formato di output (default: jsonl)
- `--label-map PATH`: Path per salvare la label map generata

## Esempio

```bash
robimb convert \
  --input data/raw/dataset.jsonl \
  --output data/processed/dataset.jsonl \
  --label-map data/processed/label_map.json
```

## Formato Output

Il comando genera un dataset processato compatibile con il training dei modelli.

## Vedi Anche

- [train.md](train.md): Training modelli con dataset convertiti
