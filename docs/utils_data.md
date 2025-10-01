# Utility per i dati

Il pacchetto `utils` è stato suddiviso in moduli specializzati:

* `utils/dataset_prep.py` ospita la logica di preprocessing vera e propria. Qui vivono `prepare_classification_dataset` (per filtrare e arricchire i JSONL grezzi), `save_datasets`, `prepare_mlm_corpus` e le funzioni che gestiscono label map e maschere ontologiche (`create_or_load_label_maps`, `build_mask_and_report`).
* `utils/sampling.py` fornisce utility leggere per leggere JSONL in `DataFrame` (`load_jsonl_to_df`) e campionare record rappresentativi (`sample_one_record_per_category`).
* `utils/registry_io.py` contiene gli helper dedicati al caricamento dei registry di proprietà e alla normalizzazione dei pacchetti di estrattori.

Il modulo storico `utils/data_utils.py` resta come shim di compatibilità e re-esporta le funzioni dalle nuove sedi, così da non interrompere eventuali import esterni.

`utils/io_utils.py` (non mostrato qui) contiene helper per scrivere file JSON/JSONL e per gestire directory temporanee. `utils/ontology_utils.py` fornisce la classe `Ontology`, funzioni per caricare l'ontologia, costruire la maschera e salvare le label map. Insieme, questi moduli creano un'infrastruttura riutilizzabile per tutte le pipeline del progetto, minimizzando la duplicazione di codice tra CLI, training e servizio.
