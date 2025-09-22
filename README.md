# BIM NLP Project

Questa repository contiene la versione riorganizzata del progetto **BIM NLP**. Il
codice è stato strutturato come pacchetto Python installabile (`src/robimb`) e
accompagnato dai dati e dalle utility necessarie per preparare i dataset,
addestrare i modelli e validarli da riga di comando.

## Struttura del repository

```
├── data/
│   ├── ontology.json                         # Ontologia gerarchica super→cat
│   ├── properties_registry_extended.json     # Registry esteso per l'estrazione proprietà
│   └── ... (altri file json/jsonl descritti più avanti)
├── outputs/                                  # Directory per artefatti generati (vuota, con .gitkeep)
├── src/
│   └── robimb/
│       ├── cli/                              # Comandi CLI convert/train/validate
│       ├── models/                           # Implementazioni LabelEmbedModel e MultiTaskBERTMasked
│       ├── training/                         # Trainer modulari per i due modelli
│       └── utils/                            # Funzioni condivise (ontologia, dati, metriche, IO)
├── README.md
└── requirements.txt (opzionale, da generare in base all'ambiente)
```

### Dati di input attesi (`data/`)

I file seguenti devono essere forniti (alcuni sono opzionali ma consigliati):

| File | Descrizione |
| ---- | ----------- |
| `train_classif.jsonl` | Dataset grezzo con campi `text`, `super_id`, `cat_id`, `uid`. |
| `label_texts_super.jsonl` | Testi descrittivi delle classi *super*. |
| `label_texts_cat.jsonl` | Testi descrittivi delle classi *cat*. |
| `ontology.json` | Ontologia che mappa ogni super alle rispettive categorie. |
| `properties_registry_extended.json` | Dizionario di proprietà per ogni `Super|Cat`. |
| `contrastive_pairs.jsonl` *(opzionale)* | Coppie per eventuale training contrastivo. |
| `run_log.jsonl` *(opzionale)* | Log storici di run o knowledge pack. |
| `done_uids.txt` *(opzionale)* | Elenco di UID da escludere/suddividere nei vari split. |

Durante la conversione vengono generati, all'interno di `outputs/`, i file:

* `train_processed.jsonl` e `val_processed.jsonl` – dataset con ID numerici `super_label` e `cat_label`.
* `label_maps.json` – mapping completo `super2id`, `cat2id`, `id2super`, `id2cat`.
* `mask_matrix.npy` – maschera S×C derivata dall'ontologia.
* `mask_report.json` – report diagnostico sulla maschera prodotta.
* `splits.json` *(se implementato)* – descrizione degli split generati.

## Installazione rapida

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt  # oppure installazione manuale di torch, transformers, datasets, pandas, numpy, scikit-learn
```

Il pacchetto può essere installato in modalità sviluppo con:

```bash
pip install -e .
```

## Utilizzo da riga di comando

### 1. Conversione dei dati

```bash
python -m robimb.cli.convert \
  --train-file data/train_classif.jsonl \
  --ontology data/ontology.json \
  --label-maps outputs/label_maps.json \
  --out-dir outputs/ \
  --make-mlm-corpus \
  --mlm-output data/mlm_corpus.txt \
  --extra-mlm data/label_texts_super.jsonl data/label_texts_cat.jsonl
```

Il comando genera i dataset preprocessati, salva la maschera ontologica e, se
richiesto, costruisce un corpus testuale per TAPT/MLM.

### 2. Training TAPT (Masked Language Modeling)

```bash
python -m robimb.training.tapt_mlm \
  data/mlm_corpus.txt \
  --model xlm-roberta-base \
  --output_dir runs/mlm_tapt
```

Lo script esegue TAPT con opzioni per whole-word masking, LLRD, congelamento e
sblocco progressivo dei layer.

### 3. Training del classificatore a label embedding

```bash
python -m robimb.cli.train label \
  --base_model runs/mlm_tapt \
  --train_jsonl outputs/train_processed.jsonl \
  --val_jsonl outputs/val_processed.jsonl \
  --label_maps outputs/label_maps.json \
  --ontology data/ontology.json \
  --label_texts_super data/label_texts_super.jsonl \
  --label_texts_cat data/label_texts_cat.jsonl \
  --out_dir runs/label_model
```

Lo script utilizza `LabelEmbedModel`, inizializza i prototipi delle classi dai
rispettivi testi e salva un pacchetto di export (`export/`) contenente pesi
`safetensors`, tokenizer, mapping e ontologia.

### 4. Training del modello gerarchico con maschera

```bash
python -m robimb.cli.train hier \
  --base_model runs/mlm_tapt \
  --train_jsonl outputs/train_processed.jsonl \
  --val_jsonl outputs/val_processed.jsonl \
  --label_maps outputs/label_maps.json \
  --ontology data/ontology.json \
  --out_dir runs/hier_model
```

Viene addestrato `MultiTaskBERTMasked` con maschera ontologica, ArcFace
opzionale e pesi di classe. Anche in questo caso il comando crea `export/` con il
modello pronto all'uso.

### 5. Validazione

```bash
python -m robimb.cli.validate \
  --model-dir runs/label_model/export \
  --test-file outputs/val_processed.jsonl \
  --label-maps outputs/label_maps.json \
  --ontology data/ontology.json \
  --output outputs/metrics.json
```

Il comando carica automaticamente il tipo corretto di modello (label o
mask-based), calcola le metriche gerarchiche e, se richiesto, esporta le
predizioni.

## Modelli

### `robimb.models.label_model.LabelEmbedModel`

* Combina un backbone Transformer con una testa di embedding L2-normalized.
* I logit sono ottenuti come similarità coseno tra embedding del testo e
  embedding delle label.
* Supporta inizializzazione da testi descrittivi, maschere ontologiche e
  salvataggio in formato Hugging Face (`save_pretrained`).

### `robimb.models.masked_model.MultiTaskBERTMasked`

* Doppia testa (super/cat) con maschera ontologica applicata su predizione e
  loss.
* Supporto ArcFace, label smoothing, pesi di classe e gestione robusta dei NaN.
* Salvataggio e caricamento compatibile con Hugging Face.

## Utility principali

* `robimb.utils.ontology_utils` – caricamento ontologia, generazione mask e
  label map.
* `robimb.utils.data_utils` – preparazione dataset, salvataggio JSONL,
  costruzione corpus MLM.
* `robimb.utils.metrics_utils` – metriche gerarchiche (accuratezza e macro-F1
  su super e cat in vista pred/gold).

## Packaging e distribuzione

Gli artefatti prodotti in `runs/<nome_run>/export` possono essere compressi e
consegnati. Ogni pacchetto contiene:

* `config.json`, `model.safetensors`/`pytorch_model.bin`, tokenizer e vocab.
* `label_maps.json`, `ontology.json`, `mask_report.json`.
* `metrics.json` con i risultati di validazione.

Per distribuire l'intero progetto è sufficiente creare un archivio contenente:

```
- src/robimb/
- data/
- outputs/ (vuota o con esempi)
- README.md
- eventuale requirements.txt / pyproject.toml
```

In ambiente di produzione è possibile installare il pacchetto, preparare i dati
con `robimb convert`, addestrare con `robimb train` e validare con `robimb
validate`, replicando l'intera pipeline descritta.

