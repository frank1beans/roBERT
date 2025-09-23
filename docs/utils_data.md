# Utility per i dati

Il modulo `utils/data_utils.py` centralizza le funzioni di IO e trasformazione dati. Oltre alle routine descritte nella pipeline di conversione, include `save_datasets`, che serializza train/val preprocessati, label map e report della maschera ontologica; `prepare_mlm_corpus`, che concatena testi e descrizioni di label rimuovendo duplicati; e `build_mask_and_report`, che restituisce sia la matrice NumPy sia un dizionario con statistiche diagnostiche.

`utils/io_utils.py` (non mostrato qui) contiene helper per scrivere file JSON/JSONL e per gestire directory temporanee. `utils/ontology_utils.py` fornisce la classe `Ontology`, funzioni per caricare l'ontologia, costruire la maschera e salvare le label map. Insieme, questi moduli creano un'infrastruttura riutilizzabile per tutte le pipeline del progetto, minimizzando la duplicazione di codice tra CLI, training e servizio.
