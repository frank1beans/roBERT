# Price Unit Implementation - Roadmap Completa

## ✅ Fatto

### 1. Dataset Preparation
- ✅ Script [normalize_price_units.py](scripts/data_prep/normalize_price_units.py)
- ✅ Normalizzazione 12 price_unit standard
- ✅ Conversione CSV → JSONL con campi `super` e `cat`
- ✅ 3,179 record pronti in `resources/data/train/price/train.jsonl`

### 2. Formato Dati Standardizzato
```json
{
  "text": "Pavimento gres...",
  "price": 35.00,
  "price_unit": "m2",
  "super": "Pavimenti",
  "cat": "Gres"
}
```

## 🔧 Da Completare

### 3. Modello - Aggiungere Price Unit Embedding

**File**: `src/robimb/models/price_regressor.py`

#### A) Aggiungi PRICE_UNIT_MAP dopo UNIT_MAP:

```python
# Price unit mappings (for price per unit)
PRICE_UNIT_MAP = {
    "cad": 0,      # € per cadauno
    "m": 1,        # € per metro lineare
    "m2": 2,       # € per metro quadrato
    "m3": 3,       # € per metro cubo
    "kg": 4,       # € per kilogrammo
    "l": 5,        # € per litro
    "h": 6,        # € per ora
    "giorno": 7,   # € per giorno
    "set": 8,      # € per set/kit
    "a_corpo": 9,  # € forfait
    "t": 10,       # € per tonnellata
    "q": 11,       # € per quintale
    "mese": 12,    # € per mese
    "g": 13,       # € per grammo
}

def get_price_unit_id(price_unit: Optional[str]) -> int:
    """Get price unit ID from price unit string."""
    if not price_unit:
        return PRICE_UNIT_MAP["cad"]
    unit_lower = price_unit.lower().strip()
    return PRICE_UNIT_MAP.get(unit_lower, PRICE_UNIT_MAP["cad"])
```

#### B) Aggiungi nel `__init__`:

```python
def __init__(
    self,
    backbone_name: str = "dbmdz/bert-base-italian-xxl-cased",
    num_properties: int = 20,
    num_units: int = 18,           # Property units
    num_price_units: int = 14,     # Price units ← NUOVO
    dropout: float = 0.1,
    use_properties: bool = True,
    property_dim: int = 64,
    unit_dim: int = 32,
    price_unit_dim: int = 16,      # ← NUOVO
    hidden_dims: List[int] = [512, 256],
    hf_token: Optional[str] = None,
):
    # ... existing code ...

    # Price unit embedding (CRITICAL!) ← NUOVO
    self.price_unit_embedding = nn.Embedding(num_price_units, price_unit_dim)

    # Update input_dim
    if use_properties:
        input_dim = hidden_size + property_dim + price_unit_dim  # ← Modificato
    else:
        input_dim = hidden_size + price_unit_dim  # ← Modificato
```

#### C) Modifica `forward`:

```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    price_unit_ids: torch.Tensor,  # ← NUOVO parametro obbligatorio
    property_ids: Optional[torch.Tensor] = None,
    property_values: Optional[torch.Tensor] = None,
    property_units: Optional[torch.Tensor] = None,
    property_mask: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    # Encode text
    outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
    cls_output = outputs.last_hidden_state[:, 0, :]
    cls_output = self.dropout(cls_output)

    # Encode price unit (CRITICAL!) ← NUOVO
    price_unit_embeds = self.price_unit_embedding(price_unit_ids)  # [batch, price_unit_dim]

    # Encode properties (if available)
    if self.use_properties and property_ids is not None:
        # ... existing property encoding ...
        combined = torch.cat([cls_output, price_unit_embeds, prop_pooled], dim=-1)  # ← Modificato
    else:
        combined = torch.cat([cls_output, price_unit_embeds], dim=-1)  # ← Modificato

    # ... rest of forward pass ...
```

### 4. Dataset - Aggiungere Price Unit

**File**: `src/robimb/training/price_trainer.py`

#### A) Aggiungi import:

```python
from ..models.price_regressor import PriceRegressor, get_unit_id, get_price_unit_id
```

#### B) Modifica `PriceDataset.__getitem__`:

```python
def __getitem__(self, idx):
    example = self.examples[idx]

    text = example["text"]
    price = example["price"]
    price_unit = example.get("price_unit", "cad")  # ← NUOVO

    # ... existing tokenization ...

    item = {
        "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
        "target": torch.tensor(log_price, dtype=torch.float),
        "price_unit_id": torch.tensor(get_price_unit_id(price_unit), dtype=torch.long),  # ← NUOVO
    }

    # ... rest of method ...
    return item
```

#### C) Modifica `train_epoch` e `evaluate`:

```python
def train_epoch(...):
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)
        price_unit_ids = batch["price_unit_id"].to(device)  # ← NUOVO

        # ... property handling ...

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            price_unit_ids=price_unit_ids,  # ← NUOVO
            property_ids=property_ids,
            property_values=property_values,
            property_units=property_units,
            property_mask=property_mask,
            targets=targets,
        )
        # ... rest of training loop ...
```

#### D) Modifica `train_price_model`:

```python
def train_price_model(args: PriceTrainingArgs) -> None:
    # ... existing code ...

    from ..models.price_regressor import UNIT_MAP, PRICE_UNIT_MAP

    model = PriceRegressor(
        backbone_name=args.backbone_name,
        num_properties=len(property_id_map),
        num_units=len(UNIT_MAP),
        num_price_units=len(PRICE_UNIT_MAP),  # ← NUOVO
        dropout=args.dropout,
        use_properties=args.use_properties,
        property_dim=args.property_dim,
        unit_dim=args.unit_dim,
        price_unit_dim=16,  # ← NUOVO (o da args)
        hidden_dims=hidden_dims,
        hf_token=hf_token,
    )

    # ... save config con num_price_units ...
```

### 5. Inferenza - Aggiungere Price Unit

**File**: `src/robimb/inference/price_inference.py`

#### A) Modifica `PriceInference.__init__`:

```python
def __init__(self, model_dir: Path, device: Optional[str] = None):
    # ... load checkpoint ...

    config = checkpoint.get("config", {})
    num_price_units = config.get("num_price_units", 14)  # ← NUOVO
    price_unit_dim = config.get("price_unit_dim", 16)    # ← NUOVO

    model = PriceRegressor(
        # ... existing params ...
        num_price_units=num_price_units,  # ← NUOVO
        price_unit_dim=price_unit_dim,    # ← NUOVO
    )
```

#### B) Modifica Pipeline.predict:

```python
def predict(self, text: str, properties: Optional[Dict[str, float]] = None,
            price_unit: str = "cad") -> Dict[str, any]:  # ← NUOVO param

    # ... tokenize text ...

    # Encode price_unit ← NUOVO
    from ..models.price_regressor import get_price_unit_id
    price_unit_id = torch.tensor([get_price_unit_id(price_unit)],
                                   dtype=torch.long, device=self.device)

    # ... encode properties ...

    outputs = self.model.forward(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        price_unit_ids=price_unit_id,  # ← NUOVO
        property_ids=property_ids,
        property_values=property_values,
        property_units=property_units,
        property_mask=property_mask,
    )

    # ... return results ...
```

### 6. CLI - Update Extract Command

**File**: `src/robimb/cli/extract.py`

Aggiorna `predict_prices`:

```python
# Nota: il comando CLI originale è stato migrato in `robimb predict price` (vedi `robimb/cli/predict.py`).
def predict_prices(
    # ... existing params ...
    price_unit_field: str = typer.Option("price_unit", "--price-unit-field"),  # ← NUOVO
    default_price_unit: str = typer.Option("cad", "--default-price-unit"),     # ← NUOVO
):
    # ... load model ...

    for record in records:
        text = record.get(text_field, "")
        properties = record.get(properties_field, {}) if use_properties else None
        price_unit = record.get(price_unit_field, default_price_unit)  # ← NUOVO

        prediction = inferencer.predict(
            text=text,
            properties=properties,
            price_unit=price_unit,  # ← NUOVO
        )
        # ... save results ...
```

## 🚀 Workflow Completo

```bash
# 1. Preparazione dati
python scripts/data_prep/normalize_price_units.py \
  resources/data/train/price/dataset.csv \
  --jsonl resources/data/train/price/train.jsonl

# 2. Training
robimb train price \
  --train-data resources/data/train/price/train.jsonl \
  --output-dir outputs/price_model \
  --use-properties true \
  --epochs 10

# 3. Inferenza
robimb predict price \
  --model-dir outputs/price_model \
  --input products.jsonl \
  --output with_prices.jsonl
```

## 📊 Esempio Completo

**Input**:
```json
{
  "text": "Pavimento gres 60x60 cm",
  "price_unit": "m2",
  "properties": {
    "dimensione_lunghezza": 600,
    "dimensione_larghezza": 600
  }
}
```

**Inferenza**:
```python
result = predictor.predict(
    text="Pavimento gres 60x60 cm",
    properties={"dimensione_lunghezza": 600, "dimensione_larghezza": 600},
    price_unit="m2"
)
# → {"price": 35.20, "log_price": 3.56, "currency": "EUR"}
```

## ✅ Checklist

- [x] Dataset normalizzato con price_unit
- [x] Script conversione CSV → JSONL
- [x] Campi `super` e `cat` corretti
- [ ] PRICE_UNIT_MAP definito nel modello
- [ ] get_price_unit_id() implementato
- [ ] Price unit embedding nel __init__
- [ ] Forward pass aggiornato con price_unit_ids
- [ ] Dataset carica price_unit
- [ ] Training loop passa price_unit_ids
- [ ] Config salva num_price_units
- [ ] Inferenza carica num_price_units
- [ ] Pipeline.predict accetta price_unit
- [ ] CLI extract supporta --price-unit-field

## 🎯 Impatto Atteso

Con `price_unit` embedding il modello imparerà che:
- **€35/m²** (pavimento) ≠ **€35/cad** (piastrella)
- **€5/m** (tubo al metro) ≠ **€5/kg** (tubo al peso)
- **€100/a_corpo** (installazione) ≠ **€100/h** (manodopera oraria)

**MAPE atteso**: Riduzione da ~20% a ~10-12% con price_unit corretto! 🎯
