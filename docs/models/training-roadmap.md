# 🎯 Roadmap Training Modelli roBERT

## 📊 Situazione Attuale

### ✅ Modelli Già Addestrati

| Modello | Tipo | Location | Status |
|---------|------|----------|--------|
| **RoBERTino** | Label classifier (edilizia) | `atipiqal/RoBERTino` (HF) | ✅ **PRONTO** - 41 super + 173 cat |
| **Span Extractor** | QA-based property extraction | `outputs/span_model/` | ✅ **PRONTO** |
| **Price Regressor** | Unit-aware price prediction | `outputs/price_model_bob/` | ✅ **PRONTO** |

### ⚠️ Modelli Implementati ma NON Addestrati

| Modello | Tipo | Codice | CLI | Status |
|---------|------|--------|-----|--------|
| **Label Classifier** | Multi-label flat classifier | ✅ `label_model.py` + `label_trainer.py` | ⚠️ Solo via `python -m` | 🔴 **NON ADDESTRATO** |
| **Hierarchical Classifier** | Masked hierarchical classifier | ✅ `masked_model.py` + `hier_trainer.py` | ⚠️ Solo via `python -m` | 🔴 **NON ADDESTRATO** |

### 📦 Dataset Disponibili

```
resources/data/train/
├── span/              # ✅ QA dataset per span extraction
├── price/             # ✅ Price dataset con price_unit
└── classification/    # ❓ DA VERIFICARE/PREPARARE
```

---

## 🚀 Piano di Training

### **FASE 1: Preparazione Dataset Classification** 🔴 PRIORITÀ ALTA

#### Obiettivo
Preparare dataset per training classificatori (label + hierarchical)

#### Comandi
```bash
# 1. Verifica se esiste già dataset classificazione
ls resources/data/train/classification/

# 2. Se non esiste, prepara dataset
robimb prepare classification \
  --input outputs/estrazione_cartongesso.jsonl \
  --output resources/data/train/classification/train.jsonl \
  --min-samples 5

# 3. Verifica formato output
head resources/data/train/classification/train.jsonl
```

**Formato Atteso**:
```json
{
  "text": "Lastra in cartongesso 120x300 cm spessore 12.5mm",
  "super_category": "Opere di muratura",
  "category": "Cartongesso"
}
```

#### Output
- `resources/data/train/classification/train.jsonl`
- `resources/data/train/classification/label_maps.json`

---

### **FASE 2: Training Label Classifier** 🟡 PRIORITÀ MEDIA

#### Obiettivo
Addestrare classificatore flat multi-label per classificazione prodotti

#### Pre-requisiti
- ✅ Dataset classification preparato
- ✅ Label maps generato
- ⚠️ Scegliere backbone model (es. `dbmdz/bert-base-italian-xxl-cased`)

#### Comandi
```bash
# Training label classifier
python -m robimb.cli.train label \
  --base-model dbmdz/bert-base-italian-xxl-cased \
  --train-jsonl resources/data/train/classification/train.jsonl \
  --val-jsonl resources/data/train/classification/val.jsonl \
  --label-maps resources/data/train/classification/label_maps.json \
  --out-dir outputs/label_classifier \
  --epochs 5 \
  --batch-size 32 \
  --proj-dim 256
```

#### Parametri Chiave
- `--base-model`: Backbone transformer (BERT italiano)
- `--proj-dim`: Dimensione embedding proiezione (default: 256)
- `--temperature`: Temperature per contrastive learning (default: 0.07)
- `--epochs`: Numero epoche (default: 5)
- `--lr-encoder`: Learning rate encoder (default: 1e-5)
- `--lr-head`: Learning rate head (default: 2e-4)

#### Output Atteso
```
outputs/label_classifier/
├── pytorch_model.bin
├── config.json
├── tokenizer files
├── label_maps.json
└── metrics.json
```

---

### **FASE 3: Training Hierarchical Classifier** 🟡 PRIORITÀ MEDIA

#### Obiettivo
Addestrare classificatore gerarchico con maschere ontologiche

#### Pre-requisiti
- ✅ Dataset classification preparato
- ✅ Label maps generato
- ✅ Ontology file (se disponibile)

#### Comandi
```bash
# Training hierarchical classifier
python -m robimb.cli.train hier \
  --base-model dbmdz/bert-base-italian-xxl-cased \
  --train-jsonl resources/data/train/classification/train.jsonl \
  --val-jsonl resources/data/train/classification/val.jsonl \
  --label-maps resources/data/train/classification/label_maps.json \
  --ontology resources/data/ontology.json \
  --out-dir outputs/hierarchical_classifier \
  --epochs 5 \
  --batch-size 32
```

#### Differenza vs Label Classifier
- **Label**: Predice flat (tutte le classi indipendenti)
- **Hierarchical**: Usa maschere ontologiche (super → category)
  - Prima predice super_category
  - Poi maschera categorie non compatibili
  - Infine predice category

#### Output Atteso
```
outputs/hierarchical_classifier/
├── pytorch_model.bin
├── config.json
├── ontology_masks.pt
└── metrics.json
```

---

### **FASE 4: Integrazione CLI Typer** 🟢 NICE TO HAVE

#### Obiettivo
Esporre `label` e `hier` come comandi Typer in `robimb train`

#### Attualmente
- ✅ Funzionano con `python -m robimb.cli.train label`
- ❌ NON esposti in `robimb train --help`

#### Azione
Aggiungere in `src/robimb/cli/train.py`:

```python
@app.command("label")
def label_command(
    base_model: str = typer.Option(..., "--base-model"),
    train_jsonl: Path = typer.Option(..., "--train-jsonl"),
    val_jsonl: Path = typer.Option(..., "--val-jsonl"),
    label_maps: Path = typer.Option(..., "--label-maps"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    epochs: int = typer.Option(5, "--epochs"),
    batch_size: int = typer.Option(32, "--batch-size"),
) -> None:
    """Train the label embedding classifier."""
    # ... implementazione

@app.command("hier")
def hier_command(...) -> None:
    """Train the hierarchical masked classifier."""
    # ... implementazione
```

#### Benefit
- Consistenza con `robimb train span` / `robimb train price`
- Migliore UX

---

## 🔍 Verifica Modelli Esistenti

### Verifica Span Model
```bash
ls outputs/span_model/
# Atteso: trainer/ folder (checkpoint)
```

### Verifica Price Model
```bash
ls outputs/price_model_bob/
# Atteso:
# - best_model.safetensors
# - property_id_map.json
# - normalizers.json
```

### Test Inferenza Span
```bash
robimb extract predict-spans \
  --model-dir outputs/span_model \
  --input test_data.jsonl \
  --output test_output.jsonl \
  --properties marchio,materiale
```

### Test Inferenza Price
```bash
robimb predict price \
  --model-dir outputs/price_model_bob \
  --input test_data.jsonl \
  --output test_output_prices.jsonl
```

---

## 📝 Checklist Completa

### Dataset Preparation
- [ ] Verifica dataset classification esistente
- [ ] Se mancante, esegui `robimb prepare classification`
- [ ] Verifica label_maps.json generato
- [ ] Split train/val (90/10 o 80/20)

### Training Label Classifier
- [ ] Scegli backbone (consigliato: `dbmdz/bert-base-italian-xxl-cased`)
- [ ] Esegui training con `python -m robimb.cli.train label`
- [ ] Monitora metrics (F1 macro, accuracy)
- [ ] Salva best checkpoint
- [ ] Testa inferenza su esempi reali

### Training Hierarchical Classifier
- [ ] Verifica ontology.json disponibile
- [ ] Esegui training con `python -m robimb.cli.train hier`
- [ ] Confronta con label classifier
- [ ] Valuta benefici maschere ontologiche

### Testing & Validation
- [ ] Test span extractor su nuovi dati
- [ ] Test price regressor su nuovi dati
- [ ] Test label classifier (se addestrato)
- [ ] Test hierarchical classifier (se addestrato)

### Documentation
- [x] Aggiorna [docs/commands/train.md](docs/commands/train.md)
- [x] Aggiorna [ARCHITECTURE.md](ARCHITECTURE.md)
- [ ] Crea esempi training in `examples/`
- [ ] Aggiungi guide troubleshooting

---

## 🎓 Note Tecniche

### Backbone Models Consigliati

| Modello | Dimensione | Velocità | Qualità | Uso |
|---------|-----------|----------|---------|-----|
| **`atipiqal/RoBERTino`** | ~450MB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **🎯 SPECIALIZZATO EDILIZIA** |
| `dbmdz/bert-base-italian-xxl-cased` | ~420MB | ⭐⭐⭐ | ⭐⭐⭐⭐ | BERT italiano generale |
| `idb-ita/gilberto-uncased-from-camembert` | ~450MB | ⭐⭐ | ⭐⭐⭐⭐⭐ | RoBERTa italiano |
| `Musixmatch/umberto-commoncrawl-cased-v1` | ~420MB | ⭐⭐⭐ | ⭐⭐⭐⭐ | BERT italiano CommonCrawl |

**Nota**: `atipiqal/RoBERTino` è **GIÀ ADDESTRATO** su dominio edilizia con 41 super-categorie e 173 categorie totali!

### Training Time (Stime)

| Modello | Dataset Size | Epochs | GPU | Tempo Stimato |
|---------|--------------|--------|-----|---------------|
| Label Classifier | 5k esempi | 5 | V100 | ~15-20 min |
| Hierarchical | 5k esempi | 5 | V100 | ~20-25 min |
| Span Extractor | 6k QA pairs | 3 | V100 | ~30-40 min |
| Price Regressor | 3k esempi | 10 | V100 | ~20-30 min |

### Hardware Requirements

**Minimo**:
- CPU: 4 cores
- RAM: 16GB
- Tempo: 4-6 ore (CPU only)

**Consigliato**:
- GPU: NVIDIA T4/V100
- VRAM: 16GB
- RAM: 32GB
- Tempo: 1-2 ore

---

## 🚧 Problemi Noti

### 1. Modello "atipiqal/RoBERTino" - CORRETTO
Il modello `atipiqal/RoBERTino` **ESISTE** su HuggingFace: https://huggingface.co/atipiqal/RoBERTino

**Note**:
- Nome corretto: `atipiqal/RoBERTino` (con R maiuscola)
- Modello italiano specializzato per edilizia
- Può essere usato come backbone per i classificatori

### 2. CLI Label/Hier Non in Typer
I comandi `label` e `hier` funzionano solo via `python -m`, non tramite `robimb train`.

**Soluzione**: Aggiungere `@app.command` in `train.py` (vedi FASE 4)

### 3. Dataset Classification Mancante
Non è chiaro se esiste dataset pronto per classification.

**Soluzione**: Eseguire `robimb prepare classification` (vedi FASE 1)

---

## 📚 Riferimenti

- [docs/commands/train.md](docs/commands/train.md) - Documentazione training
- [docs/commands/prepare.md](docs/commands/prepare.md) - Preparazione dataset
- [ARCHITECTURE.md](ARCHITECTURE.md) - Architettura sistema
- [PRICE_UNIT_IMPLEMENTATION.md](PRICE_UNIT_IMPLEMENTATION.md) - Dettagli price regressor
- [docs/PRICE_REGRESSOR.md](docs/PRICE_REGRESSOR.md) - Guida price model

---

**Prossimi Step Consigliati**:
1. ✅ Verificare se `outputs/estrazione_cartongesso.jsonl` ha campi `super_category` e `category`
2. ✅ Eseguire `robimb prepare classification` per generare dataset
3. 🔄 Decidere se addestrare label o hierarchical (o entrambi)
4. 🔄 Eseguire training e validare risultati
