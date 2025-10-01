# Gestione delle label map

Le label map definiscono la corrispondenza bidirezionale tra gli identificativi stringa di classi `super` e `cat` e gli interi utilizzati durante l'addestramento. La funzione `utils.dataset_prep.create_or_load_label_maps` incapsula il flusso: se il file JSON esiste viene letto tramite `utils.ontology_utils.load_label_maps`, altrimenti viene generato sfruttando l'ontologia (`ontology.json`).

Il file risultante contiene quattro mappe (`super2id`, `cat2id`, `id2super`, `id2cat`) e viene riutilizzato da tutto il codice. I trainer (`training/label_trainer.py` e `training/hier_trainer.py`) lo caricano per determinare `num_super`, `num_cat` e l'ID associato alla classe di fallback `#N/D`. Anche la pipeline di inferenza (`inference.predict_category._load_id2label`) lo usa per riconciliare gli indici restituiti dai modelli con le etichette leggibili.

La consistenza tra label map e ontologia è garantita dalla funzione `utils.ontology_utils.build_mask_from_ontology`, che produce anche un report diagnostico con conteggio di nodi mancanti. Qualsiasi modifica alle classi deve quindi essere riflessa in `label_maps.json` per evitare mismatch durante la validazione dei dataset e l'esecuzione dei modelli.
