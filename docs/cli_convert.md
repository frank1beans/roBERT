# Comando `robimb convert`

Il comando `convert` definito in `cli/main.py` esegue la preparazione dei dataset a partire dai file JSONL grezzi. Gli argomenti principali includono `--train-file` e `--val-file` per i sorgenti, `--ontology` per l'ontologia opzionale, `--label-maps` e `--out-dir` per le destinazioni degli artefatti. In assenza di un file di validazione viene effettuato uno split automatico controllato da `--val-split` e `--random-state`.

Opzioni aggiuntive permettono di generare un corpus TAPT (`--make-mlm-corpus`, `--mlm-output`, `--extra-mlm`), di produrre grafici e statistiche (`--reports-dir`) e di integrare la pipeline di estrazione proprietà tramite `--properties-registry` e `--extractors-pack`. Entrambe le opzioni accettano direttamente i percorsi del bundle versionato (`resources/pack/current/registry.json` e `resources/pack/current/extractors.json`, o qualsiasi `resources/pack/v*/…`). Il flag `--done-uids` consente di escludere elementi già etichettati.

Internamente il comando costruisce una `ConversionConfig` (modulo `cli/convert.py`) e invoca `run_conversion`, che restituisce percorsi agli artefatti salvati (`train_processed.jsonl`, `val_processed.jsonl`, `mask_matrix.npy`, `mask_report.json`, label map). Il risultato viene serializzato in JSON su stdout, favorendo l'integrazione con pipeline automatiche.

Eseguendo una conversione di prova con un testo di rivestimenti che contiene "spessore lastre 1,4 cm" si ottiene nella colonna `properties` il valore `"spessore_lastre_cm": 1.4`, confermando che gli slot espressi in centimetri mantengono l'unità originale senza conversioni spurie.
