# Price Regressor - Unit-Aware Price Prediction

Modello di regressione per predizione prezzi basato su transformer, **unit-aware** (consapevole dell'unità di misura del prezzo).

## 🎯 Features Chiave

1. **Unit-Aware**: Distingue €35/m² vs €35/cad
2. **Property Conditioning**: Usa proprietà estratte per migliore accuratezza
3. **Multi-Unit Support**: Gestisce automaticamente unità proprietà (mm, cm, m, kg, ecc.)
4. **Log-Scale Training**: Predizione su scala logaritmica per stabilità

## 📋 Formato Dati

### Input Training (JSONL)

```json
{
  "text": "Pavimento in gres porcellanato Florim 60x60 cm",
  "price": 35.00,
  "price_unit": "m2",
  "properties": {
    "marchio": "Florim",
    "dimensione_lunghezza": 600,
    "dimensione_larghezza": 600
  }
}
```

### Campi Obbligatori

| Campo | Tipo | Descrizione | Esempio |
|-------|------|-------------|---------|
| `text` | string | Descrizione prodotto | "Pavimento gres..." |
| `price` | float | Prezzo in EUR | `35.00` |
| `price_unit` | string | Unità di misura prezzo | `"m2"` |

### Campi Opzionali

| Campo | Tipo | Descrizione |
|-------|------|-------------|
| `properties` | object | Proprietà estratte (solo numeriche) |
| `super_category` | string | Super-categoria |
| `category` | string | Categoria |

## 🏷️ Price Units Supportate

### Standard Units

| Unit | Descrizione | Esempi |
|------|-------------|---------|
| `cad` / `pz` | € per cadauno/pezzo | Rubinetti, sanitari |
| `m` | € per metro lineare | Tubi, profili |
| `m2` | € per metro quadrato | Pavimenti, rivestimenti |
| `m3` | € per metro cubo | Calcestruzzo, ghiaia |
| `kg` | € per kilogrammo | Materiali sfusi |
| `l` | € per litro | Vernici, solventi |
| `a_corpo` | € forfait | Installazioni complete |

### Units Tempo

| Unit | Descrizione |
|------|-------------|
| `h` | € per ora (manodopera) |
| `giorno` | € per giorno |
| `mese` | € per mese |
| `anno` | € per anno |

### Altre

| Unit | Descrizione |
|------|-------------|
| `set` | € per set/kit |
| `t` | € per tonnellata |
| `q` | € per quintale |

## 🔧 Preparazione Dataset

### 1. Normalizzazione Price Units

```bash
python scripts/data_prep/normalize_price_units.py \
  resources/data/train/price/dataset.csv \
  --output resources/data/train/price/dataset_normalized.csv \
  --jsonl resources/data/train/price/train.jsonl \
  --stats
```

**Output**:
- `dataset_normalized.csv`: CSV con price_unit normalizzate
- `train.jsonl`: Dati pronti per training
- `--stats`: Mostra statistiche normalizzazione

### 2. Mapping Price Units

Lo script normalizza automaticamente varianti:

| Input | Output |
|-------|--------|
| `Cadauno`, `pezzo`, `pz`, `n` | `cad` |
| `m²`, `mq`, `metro quadrato` | `m2` |
| `m³`, `mc`, `metro cubo` | `m3` |
| `a corpo`, `corpo`, `forfait` | `a_corpo` |
| `#N/D`, ``, `null` | `cad` (default) |

## 🚀 Training

### Basic Training (solo testo + prezzo)

```bash
robimb train price \
  --train-data resources/data/train/price/train.jsonl \
  --output-dir outputs/price_model_base \
  --use-properties false \
  --epochs 10 \
  --batch-size 16
```

### Advanced Training (con proprietà)

```bash
robimb train price \
  --train-data resources/data/train/price/train_with_props.jsonl \
  --output-dir outputs/price_model_advanced \
  --use-properties true \
  --property-dim 64 \
  --unit-dim 32 \
  --price-unit-dim 16 \
  --epochs 10 \
  --batch-size 16
```

### Fine-Tuning Strategy

```bash
# Step 1: Pre-train su tutto il dataset (solo testo)
robimb train price \
  --train-data all_prices.jsonl \
  --output-dir outputs/price_base \
  --use-properties false \
  --epochs 10

# Step 2: Fine-tune con proprietà dove disponibili
robimb train price \
  --train-data prices_with_props.jsonl \
  --output-dir outputs/price_final \
  --backbone-name outputs/price_base \
  --use-properties true \
  --epochs 5 \
  --learning-rate 1e-5
```

## 🔮 Inferenza

### Python API

```python
from robimb.inference.price_inference import PriceInference

# Carica modello
predictor = PriceInference("outputs/price_model")

# Predizione semplice (solo testo)
result = predictor.predict(
    text="Pavimento gres Florim 60x60 cm",
    properties=None
)
# -> {"price": 34.50, "log_price": 3.54, "currency": "EUR"}

# Predizione con proprietà
result = predictor.predict(
    text="Pavimento gres Florim 60x60 cm",
    properties={
        "dimensione_lunghezza": 600,
        "dimensione_larghezza": 600,
        "marchio": "Florim"
    }
)
# -> {"price": 35.20, "log_price": 3.56, "currency": "EUR"}
```

### CLI

```bash
robimb predict price \
  --model-dir outputs/price_model \
  --input data/products.jsonl \
  --output data/with_prices.jsonl \
  --use-properties \
  --properties-field extracted_properties
```

## 📊 Metriche

Il modello riporta le seguenti metriche durante training/evaluation:

| Metrica | Descrizione | Buono |
|---------|-------------|-------|
| **MAPE** | Mean Absolute Percentage Error | < 15% |
| **RMSE** | Root Mean Squared Error (log-scale) | < 0.5 |
| **MAE** | Mean Absolute Error (log-scale) | < 0.3 |

## 🧠 Architettura

```
Input Text → BERT/RoBERTa Encoder → [CLS]
                                       ↓
Price Unit ID → Price Unit Embedding → Concat
                                       ↓
Properties → Property Embeddings → Property Pooling → Concat
                                       ↓
            MLP Regression Head (512→256→1)
                                       ↓
                               Log-Price Prediction
```

### Components

1. **Text Encoder**: Transformer backbone (BERT/RoBERTa)
2. **Price Unit Embedding**: `nn.Embedding(num_price_units, 16)`
3. **Property Encoder** (opzionale):
   - Property type embeddings
   - Property value encoder
   - Unit embeddings (per proprietà)
4. **Regression Head**: MLP multi-layer con dropout

## 🎓 Best Practices

### 1. Dataset Preparation

- ✅ Normalizza sempre `price_unit` prima del training
- ✅ Rimuovi outliers estremi (prezzi < 1€ o > 100k€)
- ✅ Bilancia dataset per `price_unit` (non solo m2!)
- ✅ Usa proprietà con `confidence > 0.8`

### 2. Training Strategy

**Per dataset piccoli (<5k esempi)**:
```bash
--use-properties false  # Solo testo + price_unit
--epochs 15
--learning-rate 2e-5
```

**Per dataset grandi (>10k esempi)**:
```bash
# Fase 1: Tutti i dati senza properties
--use-properties false --epochs 10

# Fase 2: Fine-tune con properties
--use-properties true --epochs 5 --lr 1e-5
```

### 3. Property Selection

Includi solo proprietà **numeriche** rilevanti per il prezzo:

✅ **Buone**:
- Dimensioni (lunghezza, larghezza, altezza)
- Peso, volume
- Portata, potenza
- Spessore

❌ **Da evitare**:
- Proprietà categoriche (usa one-hot o ignora)
- Proprietà con valori mancanti >50%
- Proprietà ridondanti (area = lung × larg)

## 📁 File Salvati

Il modello salva i seguenti file in `output_dir`:

```
outputs/price_model/
├── best_model.pt              # Best checkpoint (lowest MAPE)
├── final_model.pt             # Final checkpoint
├── property_id_map.json       # Property name → ID mapping
├── property_unit_map.json     # Property → unit mapping
├── normalizers.json           # Statistics for normalization
├── config.json                # (HF tokenizer)
├── tokenizer.json             # (HF tokenizer)
└── ...
```

## 🔍 Troubleshooting

### Predizioni sempre simili?

➡️ Verifica che `price_unit` sia diversificata nel dataset

### MAPE molto alto?

➡️ Controlla outliers, prova log-transform o rimuovi categorie anomale

### Model not learning?

➡️ Riduci `learning_rate` a 1e-5, aumenta `epochs`

## 📚 Esempi

Vedi [examples/price_regression/](../../examples/price_regression/) per esempi completi.
