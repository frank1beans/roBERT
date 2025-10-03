# Span-Based Property Extractor

Sistema intelligente per l'estrazione di proprietà da testi tecnici, che combina:
1. **Modello transformer** per identificare la porzione rilevante del testo (span)
2. **Parser/regex specifici** per estrarre il valore esatto dallo span trovato

## 🎯 Obiettivo

Invece di applicare regex/parser su tutto il testo (rischio di falsi positivi come "compensato" in "compreso e compensato"), il modello:
1. **Comprende semanticamente** quale parte del testo contiene l'informazione richiesta
2. **Estrae lo span** preciso (es. "20x20 cm" per dimensioni)
3. **Applica il parser** appropriato solo su quello span

## 📂 Struttura

```
scripts/
├── analysis/
│   └── prepare_qa_dataset.py      # Prepara dataset QA da estrazione_cartongesso.jsonl
├── training/
│   └── train_span_extractor.py    # Addestra il modello
└── inference/
    └── extract_with_spans.py      # Usa il modello per estrarre proprietà

src/robimb/models/
└── span_extractor.py               # Architettura del modello
```

## 🚀 Workflow

### 1. Preparazione Dataset

Il dataset viene creato da `outputs/estrazione_cartongesso.jsonl`, filtrando:
- Falsi positivi (es. "compensato" da "compreso e compensato")
- Estrazioni con confidence < 0.5
- Estrazioni senza span o raw text

```bash
python scripts/analysis/prepare_qa_dataset.py
```

Output: `outputs/qa_dataset/property_extraction_qa.jsonl`

Formato:
```json
{
  "context": "Miscelatore monocomando per lavabo tipo Grohe...",
  "question": "What is the brand?",
  "property_id": "marchio",
  "answers": {
    "text": ["Grohe"],
    "answer_start": [47]
  },
  "value": "Grohe",
  "confidence": 0.7,
  "source": "matcher"
}
```

### 2. Training

Addestra il modello span extractor (simile a BERT per SQuAD):

```bash
python scripts/training/train_span_extractor.py
```

**Configurazione:**
- Backbone: `dbmdz/bert-base-italian-xxl-cased`
- Batch size: 8
- Learning rate: 3e-5
- Epochs: 3
- Split: 90% train, 10% validation

**Output:**
- `outputs/span_extractor_model/best_model.pt` - Modello migliore per exact match
- `outputs/span_extractor_model/property_id_map.json` - Mapping proprietà → ID
- Tokenizer salvato nella stessa directory

**Metriche:**
- Loss (CrossEntropy su start/end positions)
- Exact Match (% span previsti esatti)

### 3. Inference

Usa il modello addestrato per estrarre proprietà:

```bash
python scripts/inference/extract_with_spans.py
```

**Pipeline:**
1. Modello trova span nel testo
2. Parser/regex estrae valore dallo span
3. Combina confidence del modello + parser

**Esempio:**
```python
from scripts.inference.extract_with_spans import SmartPropertyExtractor

extractor = SmartPropertyExtractor(
    model_dir=Path("outputs/span_extractor_model"),
    device="cuda"
)

text = "Griglia in acciaio AISI 304, dim. 20x20 cm"
results = extractor.extract_properties(text)

# Output:
# {
#   "dimensione_lunghezza": {
#     "value": 200,
#     "raw": "20x20 cm",
#     "span": (33, 41),
#     "confidence": 0.9,
#     "unit": "mm",
#     "source": "span_extractor"
#   }
# }
```

## 🧠 Architettura Modello

```
PropertySpanExtractor:
  ├── Backbone (BERT italiano)
  │   └── Encode del testo → hidden states [batch, seq_len, 768]
  │
  ├── Property Embeddings
  │   └── Embedding learnable per ogni proprietà → [batch, 768]
  │
  ├── Condizionamento
  │   └── hidden + property_embed → conditioned [batch, seq_len, 768]
  │
  └── QA Heads
      ├── Start logits [batch, seq_len]
      └── End logits [batch, seq_len]
```

## 📊 Proprietà Supportate

| Property ID | Descrizione | Parser Applicato |
|------------|-------------|------------------|
| `marchio` | Brand | Cleanup + validation |
| `materiale` | Material | Normalization |
| `dimensione_lunghezza` | Length | Dimension parser |
| `dimensione_larghezza` | Width | Dimension parser |
| `dimensione_altezza` | Height | Dimension parser |
| `tipologia_installazione` | Installation type | Normalization |
| `portata_l_min` | Flow rate | Number + unit parser |
| `classe_ei` | Fire resistance | Regex + validation |
| `classe_reazione_al_fuoco` | Fire reaction | Catalog matching |
| `presenza_isolante` | Insulation | Boolean + details |
| `stratigrafia_lastre` | Layer composition | Structured parser |

## 🔧 Personalizzazione

### Aggiungere Nuove Proprietà

1. **Aggiorna property_id_map** in `train_span_extractor.py`:
```python
property_id_map = {
    ...
    "nuova_proprieta": 20,
}
```

2. **Aggiungi parser** in `extract_with_spans.py`:
```python
elif property_id == "nuova_proprieta":
    return parse_mia_proprieta(raw_text)
```

3. **Re-train** il modello

### Migliorare Filtri False Positives

Aggiungi pattern in `prepare_qa_dataset.py`:
```python
FALSE_POSITIVE_PATTERNS = {
    "materiale": [
        (r"compreso\s+e\s+compensato", "compensato"),
        (r"tuo_pattern", "parola_da_escludere"),
    ],
}
```

## 📈 Performance Attese

Con 6886 esempi di training:
- **Exact Match** su validation: ~70-80%
- **Confidence** media: 0.85+
- **False positives** ridotti del ~90% vs regex naive

## 🔍 Debug & Analisi

Visualizza esempi del dataset:
```bash
cat outputs/qa_dataset/sample_qa_pairs.json
```

Analizza performance per proprietà:
```python
# Nel training script, aggiungi:
from collections import Counter
prop_accuracy = Counter()
for batch in val_loader:
    # ... calcola accuracy per batch["property_id"]
```

## 🚧 TODO / Miglioramenti Futuri

- [ ] Supporto multi-span (per proprietà con valori multipli)
- [ ] Ensemble con altri extractors (voting)
- [ ] Active learning per esempi difficili
- [ ] Interpretabilità (attention visualization)
- [ ] Deploy come API REST

## 📚 Riferimenti

- Architettura ispirata a [BERT for SQuAD](https://arxiv.org/abs/1810.04805)
- Dataset style: [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)
- Backbone: [dbmdz BERT Italian](https://huggingface.co/dbmdz/bert-base-italian-xxl-cased)
