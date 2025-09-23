# Comando `robimb validate`

Il comando `validate` consente di valutare un modello esportato su un dataset etichettato. Richiede percorsi espliciti a `--model-dir`, `--test-file`, `--label-maps` e, facoltativamente, a `--ontology`. Parametri aggiuntivi controllano `batch-size`, `max-length` del tokenizer, la destinazione del file di metriche (`--output`) e la serializzazione delle predizioni (`--predictions`).

L'implementazione crea un oggetto `ValidationConfig` (`cli/validate.py`) e invoca `validate_model`, che carica il modello tramite gli helper di inferenza, calcola metriche standard (precision, recall, F1 macro/micro) e genera report opzionali con matrici di confusione (`--report-dir`). Se `--output` non è specificato, le metriche vengono stampate su stdout in JSON, facilitando l'utilizzo in pipeline CI.
