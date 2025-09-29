# Documentazione di progetto

Questa cartella raccoglie una serie di note operative e di design pensate per accompagnare lo sviluppo del toolkit **robimb**. Ogni pagina approfondisce un ambito specifico del codice, dai comandi CLI ai componenti di training e inferenza. La mappa seguente elenca gli argomenti disponibili:

- [Panoramica architetturale](overview.md)
- [Pipeline di preparazione dei dataset](dataset_preparation.md)
- [Gestione delle label map](label_maps.md)
- [Maschera ontologica e report diagnostici](ontology_masks.md)
- [Estrazione proprietà testuali](property_extraction.md)
- [Modello a label embedding](label_model.md)
- [Classificatore gerarchico multi-task](masked_model.md)
- [Training del label model](training_label_model.md)
- [Training del modello gerarchico](training_hier_model.md)
- [Pre-training TAPT/MLM](training_tapt.md)
- [Comando `convert`](cli_convert.md)
- [Comando `extract`](cli_extract.md)
- [Comando `evaluate`](cli_evaluate.md)
- [Comando `pack`](cli_pack.md)
- [Script di training](cli_train.md)
- [Pipeline di inferenza end-to-end](inference_pipeline.md)
- [Servizio FastAPI](service_api.md)
- [Gestione del knowledge pack](core_knowledge_pack.md)
- [Utility per i dati](utils_data.md)
- [Utility per le metriche](utils_metrics.md)
- [Reportistica e analisi](reporting.md)
- [Changelog](changelog.md)

Le pagine sono pensate per essere lette in modo indipendente; ogni documento fornisce collegamenti al codice di riferimento e suggerimenti per estensioni future.
