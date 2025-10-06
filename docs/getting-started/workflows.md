# 🔄 Workflow Comuni

Guida pratica alle operazioni più frequenti con roBERT.

## 📋 Indice

1. [Estrazione Base](#1-estrazione-base)
2. [Estrazione da CSV](#2-estrazione-da-csv)
3. [Estrazione con LLM](#3-estrazione-con-llm)
4. [Predizione Prezzi](#4-predizione-prezzi)
5. [Training Modelli](#5-training-modelli)
6. [Batch Processing](#6-batch-processing)

---

## 1️⃣ Estrazione Base

**Obiettivo:** Estrarre proprietà da descrizioni prodotti

### Passo 1: Prepara il File di Input

Crea un file `prodotti.jsonl` (una riga JSON per prodotto):

```json
{"text": "Pavimento gres Florim 120x280 cm spessore 6mm"}
{"text": "Miscelatore Grohe Essence per lavabo, portata 5.7 l/min"}
{"text": "Lastra cartongesso Gyproc 120x300 cm sp. 12.5mm"}
```

### Passo 2: Esegui Estrazione

```bash
robimb extract properties \
  --input prodotti.jsonl \
  --output risultati.jsonl
```

### Passo 3: Visualizza Risultati

```bash
cat risultati.jsonl
```

**Output esempio:**
```json
{
  "text": "Pavimento gres Florim 120x280 cm spessore 6mm",
  "categoria": "Pavimenti in gres porcellanato",
  "properties": {
    "marchio": {"value": "Florim", "confidence": 0.92},
    "materiale": {"value": "gres_porcellanato", "confidence": 0.88},
    "dimensione_lunghezza": {"value": 1200, "unit": "mm", "confidence": 0.94},
    "dimensione_larghezza": {"value": 2800, "unit": "mm", "confidence": 0.94},
    "spessore_mm": {"value": 6, "unit": "mm", "confidence": 0.91}
  }
}
```

### Opzioni Utili

**Limita il numero di righe (per test):**
```bash
robimb extract properties \
  --input prodotti.jsonl \
  --output risultati.jsonl \
  --sample 10
```

**Estrai solo proprietà specifiche:**
```bash
robimb extract properties \
  --input prodotti.jsonl \
  --output risultati.jsonl \
  --properties marchio,materiale,dimensione_lunghezza
```

---

## 2️⃣ Estrazione da CSV

**Obiettivo:** Convertire dati da Excel/CSV e estrarre proprietà

### Passo 1: Prepara CSV

File `prodotti.csv`:
```csv
descrizione,codice
"Pavimento gres Florim 120x280 cm",PV001
"Miscelatore Grohe 5.7 l/min",MS002
```

### Passo 2: Converti CSV → JSONL

```bash
robimb convert \
  --input prodotti.csv \
  --output prodotti.jsonl
```

### Passo 3: Estrai Proprietà

```bash
robimb extract properties \
  --input prodotti.jsonl \
  --output risultati.jsonl
```

### Workflow Completo (One-Liner)

```bash
# Converti + Estrai in un comando
robimb convert --input prodotti.csv --output temp.jsonl && \
robimb extract properties --input temp.jsonl --output risultati.jsonl
```

---

## 3️⃣ Estrazione con LLM (Maggiore Precisione)

**Obiettivo:** Usare GPT-4 o altri LLM per proprietà complesse

### Requisiti

- Server LLM attivo (vedi [LLM Integration](../../examples/README.md))
- Endpoint accessibile

### Setup Server LLM (Mock per Test)

```bash
# In terminale separato
python examples/llm_server_gpt4mini.py
```

### Estrazione con LLM

```bash
robimb extract properties \
  --input prodotti.jsonl \
  --output risultati_llm.jsonl \
  --llm-endpoint http://localhost:8000/extract \
  --llm-model gpt-4o-mini
```

### Confronto Regole vs LLM

```bash
# Estrazione solo regole
robimb extract properties \
  --input prodotti.jsonl \
  --output risultati_regole.jsonl

# Estrazione con LLM
robimb extract properties \
  --input prodotti.jsonl \
  --output risultati_llm.jsonl \
  --llm-endpoint http://localhost:8000/extract

# Confronta risultati
python scripts/analysis/compare_results.py \
  risultati_regole.jsonl \
  risultati_llm.jsonl
```

---

## 4️⃣ Predizione Prezzi

**Obiettivo:** Predire prezzi basandosi su descrizione e proprietà

### Passo 1: Estrai Proprietà (se non già fatto)

```bash
robimb extract properties \
  --input prodotti.jsonl \
  --output con_proprieta.jsonl
```

### Passo 2: Predici Prezzi

```bash
robimb predict price \
  --model-dir outputs/price_model_bob \
  --input con_proprieta.jsonl \
  --output con_prezzi.jsonl
```

### Passo 3: Visualizza Risultati

```bash
cat con_prezzi.jsonl
```

**Output esempio:**
```json
{
  "text": "Pavimento gres Florim 120x280 cm",
  "properties": {...},
  "predicted_price": 35.20,
  "price_unit": "m2",
  "confidence": 0.85
}
```

### Workflow Completo Estrazione + Prezzi

```bash
# Estrazione proprietà
robimb extract properties \
  --input prodotti.jsonl \
  --output temp.jsonl

# Predizione prezzi
robimb predict price \
  --model-dir outputs/price_model_bob \
  --input temp.jsonl \
  --output risultati_finali.jsonl
```

---

## 5️⃣ Training Modelli

**Obiettivo:** Addestrare modelli personalizzati sui tuoi dati

### Training Span Extractor

**Passo 1: Prepara Dataset QA**

```bash
robimb prepare span \
  --input annotazioni.jsonl \
  --output qa_dataset.jsonl
```

**Passo 2: Training**

```bash
robimb train span \
  --train-data qa_dataset.jsonl \
  --output-dir outputs/my_span_model \
  --epochs 3 \
  --batch-size 8
```

**Passo 3: Test Modello**

```bash
robimb extract predict-spans \
  --model-dir outputs/my_span_model \
  --input test.jsonl \
  --output results.jsonl
```

### Training Price Regressor

**Passo 1: Prepara Dataset Prezzi**

```bash
robimb prepare price \
  --input prezzi_storici.csv \
  --output price_dataset.jsonl
```

**Passo 2: Training**

```bash
robimb train price \
  --train-data price_dataset.jsonl \
  --output-dir outputs/my_price_model \
  --use-properties \
  --epochs 10
```

**Passo 3: Valutazione**

```bash
robimb evaluate \
  --model-dir outputs/my_price_model \
  --test-data test_prices.jsonl \
  --output metrics.json
```

---

## 6️⃣ Batch Processing (Grandi Volumi)

**Obiettivo:** Processare migliaia di prodotti in modo efficiente

### Strategia 1: Split & Parallel

**Dividi file grande in chunks:**

```bash
# Dividi in file da 1000 righe
split -l 1000 prodotti_big.jsonl chunk_

# Processa in parallelo (Linux/macOS)
for file in chunk_*; do
  robimb extract properties --input "$file" --output "result_$file" &
done
wait

# Unisci risultati
cat result_chunk_* > risultati_completi.jsonl
```

### Strategia 2: Sampling Progressivo

**Test su campione prima di processare tutto:**

```bash
# Test su 100 prodotti
robimb extract properties \
  --input prodotti_big.jsonl \
  --output test_results.jsonl \
  --sample 100

# Analizza risultati
python scripts/analysis/extraction_results.py test_results.jsonl

# Se OK, processa tutto
robimb extract properties \
  --input prodotti_big.jsonl \
  --output final_results.jsonl
```

### Strategia 3: Usa GPU per Velocizzare

```bash
# Forza uso GPU
export CUDA_VISIBLE_DEVICES=0

# Batch size maggiore con GPU
robimb extract predict-spans \
  --model-dir outputs/span_model \
  --input prodotti_big.jsonl \
  --output results.jsonl \
  --batch-size 32
```

**Performance attese:**
- CPU: ~600ms/prodotto → 1000 prodotti in ~10 minuti
- GPU: ~150ms/prodotto → 1000 prodotti in ~2.5 minuti

---

## 🎯 Workflow Completo Produzione

**Scenario:** Importare catalogo fornitori → Database aziendale

### Fase 1: Preparazione

```bash
# Converti catalogo Excel
robimb convert \
  --input catalogo_fornitore.csv \
  --output prodotti.jsonl
```

### Fase 2: Test su Campione

```bash
# Testa su 50 prodotti
robimb extract properties \
  --input prodotti.jsonl \
  --output test.jsonl \
  --sample 50

# Verifica qualità
python scripts/analysis/extraction_results.py test.jsonl
```

### Fase 3: Estrazione Completa

```bash
# Estrai proprietà
robimb extract properties \
  --input prodotti.jsonl \
  --output con_proprieta.jsonl
```

### Fase 4: Predizione Prezzi (opzionale)

```bash
# Predici prezzi
robimb predict price \
  --model-dir outputs/price_model_bob \
  --input con_proprieta.jsonl \
  --output con_prezzi.jsonl
```

### Fase 5: Export per Database

```bash
# Converti in CSV per import DB
robimb convert \
  --input con_prezzi.jsonl \
  --output database_ready.csv
```

---

## 📊 Analisi e Validazione

### Analizza Risultati Estrazione

```bash
python scripts/analysis/extraction_results.py risultati.jsonl
```

**Metriche mostrate:**
- Proprietà estratte per categoria
- Confidence medio
- Percentuale successo
- Proprietà mancanti più comuni

### Confronta Due Estrazioni

```bash
python scripts/analysis/compare_extractions.py \
  risultati_v1.jsonl \
  risultati_v2.jsonl
```

### Valida Qualità Dataset

```bash
python scripts/testing/sanity_check.py
```

---

## 🛠️ Utilities Makefile

roBERT include un Makefile con shortcut utili:

```bash
# Estrazione regole (50 campioni)
make extract-rules SAMPLE=50

# Estrazione con LLM
make extract-llm SAMPLE=50

# Confronto regole vs LLM
make compare-llm SAMPLE=100

# Analisi risultati
make analyze OUTPUT=risultati.jsonl

# Test completo
make test
```

Vedi tutti i comandi:
```bash
make help
```

---

## 🎓 Best Practices

### ✅ Cosa Fare

1. **Testa sempre su campione** prima di processare tutto
2. **Valida risultati** con script di analisi
3. **Usa GPU** per grandi volumi
4. **Backup dati** originali prima di processing
5. **Monitora confidence** per identificare estrazioni dubbie

### ❌ Cosa Evitare

1. **Non processare file enormi** senza split
2. **Non ignorare warning** su proprietà mancanti
3. **Non usare LLM** se regole bastano (costo/tempo)
4. **Non fare training** senza dataset validato
5. **Non deploiare** senza test su dati reali

---

## 📚 Riferimenti Rapidi

### Comandi Essenziali

| Comando | Uso | Docs |
|---------|-----|------|
| `robimb convert` | CSV → JSONL | [convert.md](../commands/convert.md) |
| `robimb extract` | Estrazione proprietà | [extract.md](../commands/extract.md) |
| `robimb predict` | Predizione prezzi | [predict.md](../commands/predict.md) |
| `robimb train` | Training modelli | [train.md](../commands/train.md) |
| `robimb prepare` | Preparazione dataset | [prepare.md](../commands/prepare.md) |

### Script Utili

| Script | Scopo |
|--------|-------|
| `scripts/analysis/extraction_results.py` | Analizza qualità estrazione |
| `scripts/analysis/dataset_basic.py` | Statistiche dataset |
| `scripts/testing/sanity_check.py` | Test sistema |

---

## 🆘 Problemi Comuni

### Estrazione Lenta
**Soluzione:** Usa `--sample` per test, GPU per produzione

### Proprietà Mancanti
**Soluzione:** Verifica formato input, usa LLM per casi complessi

### Confidence Bassa
**Soluzione:** Training su dataset specifico, o validazione manuale

### Out of Memory
**Soluzione:** Riduci batch size, split file grandi

---

**Precedente:** [Installazione](installation.md) | **Prossimo:** [Comandi CLI](../commands/overview.md)
