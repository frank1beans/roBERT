# BIM NLP Project

Questa repository contiene la versione riorganizzata del progetto **BIM NLP**. Il
codice è stato strutturato come pacchetto Python installabile (`src/robimb`) e
accompagnato dai dati e dalle utility necessarie per preparare i dataset,
addestrare i modelli e validarli da riga di comando.

## Struttura del repository

```
├── data/                                     # Directory di lavoro per file utente (vuota)
├── outputs/                                  # Directory per artefatti generati (vuota, con .gitkeep)
├── src/
│   └── robimb/
│       ├── cli/                              # Comandi CLI convert/train/validate
│       ├── extraction/                       # Motore regex, normalizzatori e risorse JSON
│       │   └── resources/                    # Accesso al knowledge pack distribuito
│       ├── models/                           # Implementazioni LabelEmbedModel e MultiTaskBERTMasked
│       ├── training/                         # Trainer modulari per i due modelli
│       └── utils/                            # Funzioni condivise (ontologia, dati, metriche, IO)
├── pyproject.toml                         # Metadata del pacchetto e dipendenze
├── README.md
└── tests/                                 # Smoke test per la CLI Typer
```

### Dati di input attesi (`data/`)

I file seguenti devono essere forniti (alcuni sono opzionali ma consigliati):

| File | Descrizione |
| ---- | ----------- |
| `train_classif.jsonl` | Dataset grezzo con campi `text`, `super_id`, `cat_id`, `uid`. |
| `label_texts_super.jsonl` | Testi descrittivi delle classi *super*. |
| `label_texts_cat.jsonl` | Testi descrittivi delle classi *cat*. |
| `ontology.json` | Ontologia che mappa ogni super alle rispettive categorie. |
| `properties_registry.json` *(opzionale)* | Dizionario personalizzato di proprietà per ogni `Super|Cat`. |
| `contrastive_pairs.jsonl` *(opzionale)* | Coppie per eventuale training contrastivo. |
| `run_log.jsonl` *(opzionale)* | Log storici di run o knowledge pack. |
| `done_uids.txt` *(opzionale)* | Elenco di UID da escludere/suddividere nei vari split. |

### Asset di estrazione proprietà

Gli asset per l'estrazione automatica delle proprietà, insieme a registry, mappe di categoria, validatori e template descrittivi, sono raccolti in un unico knowledge pack distribuito in `pack/current/pack.json`. La funzione `robimb.extraction.resources.load_default()` restituisce direttamente la sezione `extractors` presente nel pack, mentre `robimb.registry.load_pack()` carica l'intero bundle quando serve accedere anche a registry e template.

Gli script `robimb convert` e la pipeline di inferenza utilizzano questo stesso file come punto di ingresso predefinito. È comunque possibile fornire un pack alternativo via CLI (`--extractors-pack` o `--properties-registry`) indicando un JSON con struttura analoga oppure un knowledge pack completo contenente la chiave `extractors`.

Durante la conversione vengono generati, all'interno di `outputs/`, i file:

* `train_processed.jsonl` e `val_processed.jsonl` – dataset con ID numerici `super_label` e `cat_label`.
* `label_maps.json` – mapping completo `super2id`, `cat2id`, `id2super`, `id2cat`.
* `mask_matrix.npy` – maschera S×C derivata dall'ontologia.
* `mask_report.json` – report diagnostico sulla maschera prodotta.
* `reports/` – grafici e statistiche descrittive su train/val (distribuzioni label, lunghezza testi, ecc.).
* `splits.json` *(se implementato)* – descrizione degli split generati.

## Prerequisiti

Il progetto richiede **Python 3.9+** e **PyTorch 2.x**. Gli esperimenti sono
stati validati con `torch 2.1`, pertanto si consiglia di utilizzare la stessa
minor release per evitare incompatibilità con `transformers` e con gli script
di training distribuiti.

## Installazione rapida

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

L'installazione precedente crea la console `robimb` e installa le dipendenze
necessarie (inclusi `pytest` e gli extra Typer per i test). Per un'installazione
runtime minimale è sufficiente `pip install .`.

## Utilizzo da riga di comando

La console unificata `robimb` espone tutti i comandi principali. Verificare la
versione installata con:

```bash
robimb --version
```

### 1. Conversione dei dati

```bash
robimb convert \
  --train-file data/train_classif.jsonl \
  --ontology data/ontology.json \
  --label-maps outputs/label_maps.json \
  --out-dir outputs/ \
  --make-mlm-corpus \
  --mlm-output data/mlm_corpus.txt \
  --extra-mlm data/label_texts_super.jsonl --extra-mlm data/label_texts_cat.jsonl
```

Il comando genera i dataset preprocessati, salva la maschera ontologica e, se
richiesto, costruisce un corpus testuale per TAPT/MLM.

### 2. Estrazione del knowledge pack

```bash
robimb extract \
  --registry pack/current/registry.json \
  --extractors pack/current/extractors.json \
  --out-dir data/properties
```

Espande un knowledge pack monolitico nella struttura a cartelle (`registry/` ed
`extractors/`) utilizzata dal progetto.

### 3. Creazione del knowledge pack

```bash
robimb pack \
  --properties-root data/properties \
  --out-registry pack/out/registry.json \
  --out-extractors pack/out/extractors.json
```

Ricompone le cartelle delle proprietà in due file JSON pronti per la
distribuzione o la pubblicazione.

### 4. Valutazione del modello

```bash
robimb evaluate \
  --model-dir runs/label_model/export \
  --test-file outputs/val_processed.jsonl \
  --label-maps outputs/label_maps.json \
  --ontology data/ontology.json \
  --output outputs/metrics.json
```

Il comando carica automaticamente il tipo corretto di modello (label o
mask-based), calcola le metriche gerarchiche e, se richiesto, esporta le
predizioni.

### Script di training e TAPT

I flussi di addestramento rimangono disponibili come script dedicati:

```bash
python -m robimb.cli.train label ...
python -m robimb.cli.train hier ...
python -m robimb.training.tapt_mlm ...
```

Questa modalità preserva tutti gli argomenti avanzati dei tool originali senza
appesantire il router Typer principale.

### 5. Reportistica e visualizzazioni

La pipeline genera automaticamente una reportistica visuale pensata per analizzare
sia i dataset in ingresso sia le prestazioni in uscita:

* Durante `robimb convert` viene popolata la cartella `reports/` (configurabile con
  `--reports-dir`), contenente:
  * istogrammi delle lunghezze testuali per train e validation;
  * distribuzioni delle classi *super* e *cat* (grafici a barre ordinati);
  * un file `dataset_summary.json` con statistiche descrittive (conteggi, medie,
    percentile 95) utili per monitorare sbilanciamenti e anomalie.
* Con `robimb evaluate` si possono produrre artefatti diagnostici passando
  `--report-dir outputs/eval_reports`:
  * matrici di confusione normalizzate (super e cat) renderizzate con seaborn;
  * `evaluation_prediction_report.json` con classification report dettagliati e
    l'elenco delle principali coppie confuse.

I grafici sono realizzati con `matplotlib`/`seaborn` in modalità headless (backend
`Agg`), quindi sono generabili anche su macchine senza interfaccia grafica. Le
sezioni di reportistica possono essere archiviate insieme agli artefatti di run
per alimentare dashboard esterne o documentazione interna del progetto BIM NLP.

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

