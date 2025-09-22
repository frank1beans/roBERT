# roBERT modular pipelines

Questo repository contiene ora un progetto Python strutturato come package (`robert/`) che
raccoglie la logica comune per il training e l'inferenza dei modelli gerarchici
utilizzati per classificare le lavorazioni edili e stimarne le proprietà
collegate.

## Struttura del package

```
robert/
├── __init__.py                 # Esporta le API principali
├── config.py                   # Dataclass di configurazione per model, training, inference e pipeline
├── data/
│   └── ontology.py             # Utility per ontologia, label maps e matrici di mascheramento
├── models/
│   ├── masked.py               # Implementazione del modello MultiTaskBERTMasked (ex masked_model.py)
│   └── label.py                # Implementazione del LabelEmbedModel (ex label_model.py)
├── pipelines/
│   ├── inference.py            # Pipeline di inferenza con estrazione proprietà
│   └── training.py             # Pipeline di training per MLM e Label Embedding
└── properties/
    ├── registry.py             # Classi per gestire registri e slot di proprietà
    └── extractors.py           # Estrattori regex basati sul registro
```

I vecchi file `masked_model.py` e `label_model.py` sono rimasti come wrapper di
retro-compatibilità e reindirizzano alle nuove posizioni.

## Dataset, ontologia e registry proprietà

Gli helper in `robert.data.ontology` permettono di:

* caricare ontologie (`load_ontology`) e label-map storiche (`load_label_maps`)
* costruire matrici di mascheramento consistenti con la gerarchia (`build_mask`)
  con la possibilità di ottenere report diagnostici sulla copertura.

La gestione delle proprietà è stata resa modulare tramite:

* `PropertyRegistry` per rappresentare l'intero registro (incluso `_schema`)
* `PropertyGroup` per ogni coppia `Super|Cat` con priorità, slot e pattern
* `PropertySlot` per descrivere e normalizzare ciascuna proprietà
* `RegexPropertyExtractor` per estrarre valori dal testo in funzione del gruppo
  previsto, restituendo risultati normalizzati e ordinati secondo la priorità
  definita nel registro.

Il registro esteso (`properties_registry_extended.json`) può quindi essere
caricato, manipolato (merge/append) e nuovamente serializzato in modo semplice.

## Pipeline di training

`robert.pipelines.training` espone due classi:

* `MaskedMLMTrainingPipeline` — per addestrare `MultiTaskBERTMasked` a partire da
  file JSONL con chiavi configurabili (`text`, `super`, `cat`).
* `LabelEmbeddingTrainingPipeline` — per addestrare il modello a label embedding
  con logica analoga.

Entrambe ereditano da una base comune che:

1. carica le label maps e, opzionalmente, l'ontologia per ricostruire la mask;
2. effettua la normalizzazione dei record e la costruzione dei dataset Hugging
   Face `Dataset`;
3. applica la tokenizzazione con la lunghezza massima richiesta;
4. costruisce il modello con i parametri dichiarati in `PipelineConfig`.

La `PipelineConfig` combina le dataclass `ModelConfig` e `TrainingConfig` per
centralizzare gli iper-parametri, con la possibilità di aggiungere argomenti
extra (es. nomi di campo del dataset). Entrambe le pipeline restituiscono un
`Trainer` Hugging Face già configurato, pronto per la chiamata a `train()`.

Gli script CLI storici (`train_hier_masked.py` e `train_label.py`) sono stati
aggiornati per sfruttare le nuove utility di ontologia/label maps mantenendo le
interfacce originali.

## Pipeline di inferenza

La classe `InferencePipeline` consente di:

1. caricare il modello gerarchico e, opzionalmente, quello a label embedding;
2. gestire label maps e mask coerenti con l'ontologia fornita;
3. applicare in batch la predizione su una lista di testi;
4. estrarre automaticamente le proprietà dal testo in base al gruppo predetto
   sfruttando `RegexPropertyExtractor`.

Il risultato (`PredictionOutput`) include ID, etichette e score di `super` e
`cat`, oltre alle proprietà rilevate (con relativa traccia del pattern usato).

## Utilizzo rapido

```python
from pathlib import Path

from robert import (
    ModelConfig, TrainingConfig, PipelineConfig,
    MaskedMLMTrainingPipeline, InferencePipeline,
    InferenceConfig,
)

model_cfg = ModelConfig(name_or_path="dbmdz/bert-base-italian-xxl-uncased")
train_cfg = TrainingConfig(epochs=3, batch_size=32, output_dir=Path("outputs/mlm"))
pipeline_cfg = PipelineConfig(model=model_cfg, training=train_cfg)

pipeline = MaskedMLMTrainingPipeline(
    pipeline_cfg,
    label_maps_path="label_maps.json",
    ontology_path="ontology.json",
)
trainer = pipeline.fit("train.jsonl", "val.jsonl")
trainer.train()

inf_cfg = InferenceConfig(
    masked_model_path=Path("outputs/mlm/checkpoint-best"),
    label_maps_path=Path("label_maps.json"),
    ontology_path=Path("ontology.json"),
    properties_registry_path=Path("properties_registry_extended.json"),
)
inference = InferencePipeline(inf_cfg)
results = inference.predict(["Testo di descrizione lavorazione..."])
print(results[0].properties)
```

Questa struttura modulare semplifica l'integrazione di nuovi modelli o estrattori
in futuro, garantendo riuso del codice esistente e una gestione uniforme di
ontologie, registry di proprietà e pipeline di addestramento/inferenza.
