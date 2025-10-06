# robimb prepare - Preparazione Dataset

Prepara dataset per training dei modelli di span extraction, price regression e classification.

## Sintassi

```bash
robimb prepare [SUBCOMMAND] [OPTIONS]
```

## Subcomandi

### Classification Dataset

Prepara dataset per classificazione (text → super_category, category).

```bash
robimb prepare classification \
  --input data/raw.jsonl \
  --output data/classification.jsonl \
  --min-samples 5
```

**Formato Output**:

```json
{
  "text": "Pavimento gres Florim 60x60 cm",
  "super_category": "Pavimentazioni",
  "category": "Gres porcellanato"
}
```

### Span Extraction Dataset

Prepara dataset per span extraction (estrae proprietà con LLM, poi crea formato QA).

```bash
robimb prepare span \
  --input data/raw.jsonl \
  --output-dirrobimb prepare span  --input resources\data\train\span\raw\train_for_extract.jsonl  --output data/qa_data.jsonl  --llm-endpoint http://localhost:11434/extract data/qa_data.jsonl \
  --llm-endpoint http://localhost:8000/extract \
  --properties marchio,materiale,dimensione_lunghezza
```

**Formato Output**:

```json
{
  "text": "Pavimento gres Florim 60x60 cm",
  "property": "marchio",
  "answer_text": "Florim",
  "answer_start": 16,
  "answer_end": 22,
  "context": "Pavimento gres Florim 60x60 cm"
}
```

### Price Regression Dataset

Prepara dataset per price prediction (text → price, price_unit).

```bash
robimb prepare price \
  --input data/raw.csv \
  --output data/price_data.jsonl \
  --normalize-units
```

**Formato Output**:

```json
{
  "text": "Pavimento gres Florim 60x60 cm",
  "price": 35.50,
  "price_unit": "m2",
  "properties": {
    "dimensione_lunghezza": 600,
    "dimensione_larghezza": 600
  }
}
```

### All Datasets

Prepara tutti i dataset in un solo comando.

```bash
robimb prepare all \
  --input data/raw.jsonl \
  --output-dir data/prepared/ \
  --llm-endpoint http://localhost:8000/extract
```

**Output**:

- `data/prepared/classification.jsonl`
- `data/prepared/span_qa.jsonl`
- `data/prepared/price.jsonl`

## Opzioni Comuni

### Input/Output

- `--input PATH`: File di input (JSONL o CSV)
- `--output PATH`: File di output (per subcomandi specifici)
- `--output-dir PATH`: Directory di output (per `all`)

### Span Extraction

- `--llm-endpoint URL`: Endpoint LLM per estrazione proprietà
- `--llm-model NAME`: Modello LLM (default: gpt-4o-mini)
- `--properties LIST`: Lista proprietà da estrarre (comma-separated)

### Price Regression

- `--normalize-units`: Normalizza price_unit (m² → m2, cadauno → cad, etc.)
- `--include-properties`: Includi proprietà estratte nel dataset

### Classification

- `--min-samples INT`: Minimo esempi per categoria (default: 5)
- `--balance`: Bilancia dataset per categoria

## Workflow Tipico

### 1. Prepara Span Dataset

```bash
# Step 1: Estrai proprietà con LLM (crea annotazioni)
robimb prepare span \
  --input data/raw_descriptions.jsonl \
  --output data/span_qa.jsonl \
  --llm-endpoint http://localhost:8000/extract \
  --properties marchio,materiale,dimensione_lunghezza,dimensione_larghezza,spessore

# Step 2: Train span model
robimb train span \
  --train-data data/span_qa.jsonl \
  --output-dir outputs/span_model
```

### 2. Prepara Price Dataset

```bash
# Step 1: Normalizza e converti CSV → JSONL
robimb prepare price \
  --input data/price_raw.csv \
  --output data/price_data.jsonl \
  --normalize-units \
  --include-properties

# Step 2: Train price model
robimb train price \
  --train-data data/price_data.jsonl \
  --output-dir outputs/price_model \
  --use-properties
```

### 3. Prepara Tutti i Dataset

```bash
# Tutto in un comando
robimb prepare all \
  --input data/complete_dataset.jsonl \
  --output-dir data/prepared/ \
  --llm-endpoint http://localhost:8000/extract \
  --normalize-units

# Output:
# data/prepared/classification.jsonl
# data/prepared/span_qa.jsonl
# data/prepared/price.jsonl
```

## Requisiti Dataset Input

### Per Span Extraction

Campo obbligatorio: `text` (descrizione prodotto)

### Per Price Regression

Campi obbligatori: `text`, `price`, `price_unit`

### Per Classification

Campi obbligatori: `text`, `super_category`, `category`

## Note

- **LLM Endpoint**: Per span extraction, è consigliato usare un LLM per annotare le proprietà
- **Normalizzazione**: `--normalize-units` converte varianti (m², mq → m2; cadauno, pz → cad)
- **Qualità**: Verifica sempre la qualità del dataset preparato prima del training

## Vedi Anche

- [train.md](train.md): Training modelli
- [extract.md](extract.md): Estrazione proprietà
- [Price Regressor](../PRICE_REGRESSOR.md): Dettagli price regression
