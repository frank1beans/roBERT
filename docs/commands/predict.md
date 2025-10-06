# robimb predict

CLI dedicata alle attività di inferenza basate sui modelli addestrati (classificazione categorie, estrazione proprietà e regressione prezzi).

## Sintassi Generale

```bash
robimb predict [category|properties|price] [OPZIONI]
```

Le opzioni comuni includono:
- `--text`: elabora un singolo testo passato da linea di comando
- `--input`: percorso a file JSONL con più record
- `--output`: file JSONL di destinazione (default: stdout)
- `--pretty`: formatta l'output in modo leggibile per la modalità single-text
- `--progress/--no-progress`: abilita/disabilita la progress bar

## Comandi Disponibili

### category
Predice la categoria BIM di un testo usando modelli `LabelEmbed` o classificatori Hugging Face standard.

```bash
robimb predict category \
  --model-dir outputs/robertino \
  --input data/products.jsonl \
  --output outputs/categories.jsonl \
  --top-k 5
```

Opzioni rilevanti:
- `--backend`: forza `label-embed` o `sequence-classifier` (default: `auto`)
- `--label-map`: file JSON con mapping `id -> label` per i modelli HF privi di metadata
- `--include-scores`: include logits e probabilità nell'output

### properties
Esegue l'estrazione di proprietà tramite il modello span-based, con supporto a batch e filtri opzionali.

```bash
robimb predict properties \
  --model-dir outputs/span_model \
  --input data/products.jsonl \
  --output outputs/properties.jsonl \
  --batch-size 16
```

Opzioni rilevanti:
- `--property-id`: ripetibile, limita l'estrazione alle proprietà indicate
- `--raw-spans`: disattiva i parser di dominio restituendo il testo grezzo
- `--batch-size`: dimensione dei batch in modalità file (default: 8)

### price
Calcola il prezzo utilizzando il regressore addestrato; supporta sia singoli testi che dataset JSONL.

```bash
robimb predict price \
  --model-dir outputs/price_model \
  --input data/products.jsonl \
  --output outputs/prices.jsonl \
  --use-properties
```

Opzioni rilevanti:
- `--properties-field`: campo contenente le proprietà estratte (default: `properties`)
- `--properties-json`: JSON inline delle proprietà per la modalità single-text
- `--price-unit-field` / `--default-price-unit`: gestiscono l'unità di prezzo
- `--no-properties`: ignora completamente le proprietà anche se presenti

## Suggerimenti

- Impostare `HF_TOKEN` nell'ambiente per scaricare modelli privati.
- Utilizzare `--output-field` per controllare il nome del campo di output aggiunto a ciascun record.
- Combinare `robimb predict category` e `robimb predict properties` in pipeline personalizzate prima di chiamare `robimb predict price`.
