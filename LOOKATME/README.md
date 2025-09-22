
# roBIMB – Project Scaffold (Step 1/10)
Generato: 2025-09-21T13:10:26.282667Z

Questo è lo **scheletro** del progetto. Nei prossimi step andremo a implementare:
1. Conversione dataset (jsonl → train/val/test, label index)  
2. Pack/ontologia: schema + loader + validazione  
3. Extractors engine (regex + normalizer plug-in)  
4. Validators engine (range/presenza/contesto)  
5. Training (TAPT/MLM, label, gerarchico) con modelli **atipiqal/BOB** e **atipiqal/roBERTino**  
6. Inference pipeline (cat + proprietà + descrizione)  
7. API FastAPI + Docker  
8. Valutazione e calibrazione  
9. CI/CD e manifest pack  
10. Documentazione completa

Avvio rapido (tra qualche step): `pip install -e .` e CLI `robimb ...`.

## Step 2 — Converter (dataset)
Esempio:
```bash
pip install -e .
robimb convert --src data/raw/keynote_jsonl/tuo.jsonl --out data --seed 42 --lowercase
```
Output:
- `data/processed/classification/{super,cat}/{train,val,test}.jsonl`
- `data/processed/hier/{train,val,test}.jsonl`
- `data/processed/mlm/train.txt`
- `data/metadata/label_index/{super.json,cat.json}`
- `data/metadata/splits/split_seed42.json`
- `data/metadata/stats.json`



## Step 3 — Pack validator
Valida la struttura del **Knowledge Pack** contro schemi JSON interni.
```bash
pip install -e .
robimb pack-validate --pack pack/current/pack.json
```
Output: JSON con `ok` globale e report per ciascun file (errori, sha256, size).



## Step 4 — Extractors engine
- Normalizzatori **built-in**: `comma_to_dot`, `to_float`, `to_int`, `EI_from_any`, `dims_join`, `if_cm_to_mm`, `collect_many`, `unique_list`, `lower/upper/strip`.
- Normalizzatori **dinamici**: `map_enum:<name>` che usa mappe definite in `extractors.normalizers.<name>`.
- CLI dry-run:
```bash
robimb extract --pack pack/current/pack.json --text "Muratura in laterizio forato sp 25 cm REI 120; Rw 54 dB; Uw=1,3; 60x60"
```
- Test rapidi:
```bash
pytest -q
```



## Step 5 — Validators engine
Regole supportate:
- `requires`: lista di proprietà obbligatorie (es. per un profilo tipo “masonry”)
- `range`: `{ "prop": "geo.spessore_elemento", "min": 30, "max": 2000 }`
- `enum`: `{ "prop": "opn.tenuta_aria_EN12207", "in": ["0","1","2","3","4"] }`
- `regex`: `{ "prop":"idn.descrizione", "pattern":"\\bmuratura\\b" }`
- `if` + `assert` (espressioni sicure): env include `exists(p)`, `val(p)`, `num(p)`, `category`, `context`, `re`.
- `when`: filtra l’applicazione della regola (per `category`, `type_profile`, `regex`, `context`).

CLI:
```bash
# 1) Validazione da props inline
robimb validate --pack pack/current/pack.json --category "Murature in laterizio"   --props '{"geo.spessore_elemento": 250, "frs.resistenza_fuoco_EI": "EI 120"}'

# 2) Validazione estraendo dal testo con extractors
robimb validate --pack pack/current/pack.json --category "Murature in laterizio"   --text "Muratura in laterizio forato sp 25 cm REI 120"
```
Output: elenco `issues` (code, message, severity, property_id, rule_id) e le `props` verificate.



## Step 6 — Trainer (TAPT/MLM + Label + Hier)
Modelli di riferimento:
- **TAPT/MLM** → `atipiqal/BOB`
- **Label/Hier** → `atipiqal/roBERTino`

### TAPT/MLM
Config (esempio `configs/tapt.json`):
```json
{
  "model_name": "atipiqal/BOB",
  "train_corpus": "data/processed/mlm/train.txt",
  "save_dir": "runs/tapt",
  "block_size": 128,
  "mlm_probability": 0.15,
  "epochs": 3,
  "batch_size": 16,
  "lr": 5e-5,
  "seed": 42
}
```
Comando:
```bash
robimb train --task tapt --config configs/tapt.json
```

### Classificatore (CAT o SUPER)
Config (esempio `configs/label.json`):
```json
{
  "model_name": "atipiqal/roBERTino",
  "train_path": "data/processed/classification/cat/train.jsonl",
  "val_path": "data/processed/classification/cat/val.jsonl",
  "label_index": "data/metadata/label_index/cat.json",
  "save_dir": "runs/label",
  "max_length": 256,
  "batch_size": 16,
  "epochs": 5,
  "lr": 3e-5,
  "seed": 42,
  "llrd_decay": 0.9,
  "freeze_n_layers": 0
}
```
Comando:
```bash
robimb train --task label --config configs/label.json
```

### Gerarchico (SUPER + CAT insieme)
Config (esempio `configs/hier.json`):
```json
{
  "model_name": "atipiqal/roBERTino",
  "train_path": "data/processed/hier/train.jsonl",
  "val_path": "data/processed/hier/val.jsonl",
  "super_index": "data/metadata/label_index/super.json",
  "cat_index":   "data/metadata/label_index/cat.json",
  "save_dir": "runs/hier",
  "max_length": 256,
  "batch_size": 16,
  "epochs": 5,
  "lr": 3e-5,
  "seed": 42,
  "llrd_decay": 0.9,
  "freeze_n_layers": 0,
  "alpha_super": 0.3,
  "alpha_cat": 0.7
}
```
Comando:
```bash
robimb train --task hier --config configs/hier.json
```
Note: il modello gerarchico usa due teste lineari sul backbone e una loss combinata.



## Step 7 — Inference end‑to‑end + Calibrazione
### Predizione completa
```bash
robimb predict   --pack pack/current/pack.json   --text "Muratura in laterizio forato sp 25 cm REI 120"   --model runs/label \            # oppure atipiqal/roBERTino addestrato
  --label-index data/metadata/label_index/cat.json   --topk 5   --calibrator runs/label/calibrator.json   # opzionale
```
Output JSON: `category` (top1 con score), `topk` (lista), `properties` (estratte con regex), `issues` (validator), `description` (template).

### Calibrazione (temperature scaling)
```bash
robimb calibrate   --model runs/label   --label-index data/metadata/label_index/cat.json   --val data/processed/classification/cat/val.jsonl   --out runs/label/calibrator.json
```
La calibrazione regola le probabilità (softmax) per ridurre l'over/under‑confidence sulle classi.



## Step 8 — API FastAPI + Docker
### Avvio locale (venv)
```bash
uvicorn robimb.service.app:app --host 0.0.0.0 --port 8080
# health
curl http://localhost:8080/health
# predict
curl -s http://localhost:8080/predict -X POST -H "Content-Type: application/json"   -d '{"text":"Muratura in laterizio forato sp 25 cm REI 120", "topk": 5}' | jq
```

### Docker (CPU)
```bash
docker build -f docker/Dockerfile.cpu -t robimb/api:latest .
docker run --rm -it -p 8080:8080 \
  -e ROBIMB_PACK=/app/pack/current/pack.json \
  -v $PWD/pack:/app/pack:ro -v $PWD/data:/app/data:ro -v $PWD/runs:/app/runs:ro \
  robimb/api:latest
```

### Docker Compose
```bash
cp .env.example .env
docker compose up --build -d
curl http://localhost:8080/health
```

Endpoints:
- `GET /health` → stato, versioni, env
- `POST /predict` → input `{"text": "...", "topk": 5}` ⇒ categoria, proprietà, issues, descrizione

Generato: 2025-09-21T14:06:49.835462Z



## Step 9 — Evaluation & Calibration (con grafici)
Valuta un classificatore su uno **split** (val/test), genera **metriche**, **grafici** e un **report HTML**.

### Esempio
```bash
robimb eval   --model runs/label   --label-index data/metadata/label_index/cat.json   --data data/processed/classification/cat/test.jsonl   --out runs/eval/cat_test   --calibrator runs/label/calibrator.json
```

### Output
- `metrics.json` (accuracy, f1_macro/micro, top-k@1/3/5, NLL, Brier, ECE/MCE, PR/ROC macro)
- `predictions.jsonl` (testo, true/pred, score, top5) — utile per **active learning**
- Figure PNG (una per grafico, **matplotlib senza colori impostati**):
  - `confusion_matrix.png`
  - `reliability.png` (diagramma affidabilità)
  - `confidence_hist.png`
  - `topk_accuracy.png`
  - `risk_coverage.png`
  - `pr_macro.png`, `roc_macro.png`
- `report.html` con **riepilogo** e **immagini** (standalone)

Generato: 2025-09-21T14:13:13.799816Z



## Step 10 — CI/CD, Pack manifest, Makefile, Active Learning
### Costruzione Pack
```bash
robimb pack-build --src pack/v0 --out pack/v1 --set-current
robimb pack-validate --pack pack/current/pack.json
```
Genera `pack/v1/manifest.json` con SHA256 e aggiorna `pack/current/pack.json` alla nuova versione.

### Active Learning export
Dopo `robimb eval`, esporta i sample più incerti:
```bash
robimb al-export --preds runs/eval/cat_test/predictions.jsonl --out runs/eval/cat_test/active_learning.csv --k 500 --strategy margin --per-class 10
```
- **Strategie**: `margin` (top1-top2), `entropy`, `leastconf` (1-top1).  
- CSV con: `text; true_label; pred_label; score; uncertainty; top5_json; misclassified`.

### Makefile
Comandi rapidi: `make install | convert | tapt | mlm | label | hier | predict | calibrate | eval | api | docker-build | docker-up | pack-validate | pack-build`.

### CI (GitHub Actions)
- Workflow `.github/workflows/ci.yml`:
  - installazione pacchetto
  - **smoke test** pack loader
  - validazione pack

**FINE — progetto pronto per produzione.**  
Generato: 2025-09-21T14:18:11.996231Z
