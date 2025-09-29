# Comando `robimb evaluate`

Il comando `evaluate` consente di valutare un modello esportato su un dataset etichettato. Richiede percorsi espliciti a `--model-dir`, `--test-file`, `--label-maps` e, facoltativamente, a `--ontology`. Parametri aggiuntivi controllano `batch-size`, `max-length` del tokenizer, la destinazione del file di metriche (`--output`) e la serializzazione delle predizioni (`--predictions`).

Internamente la CLI costruisce un oggetto `EvaluationConfig` (`cli/evaluate.py`) e invoca `evaluate_model`, che carica automaticamente il tipo corretto di modello (label embedding o gerarchico) tramite gli helper di inferenza. La funzione calcola le metriche gerarchiche (precision, recall, F1 macro/micro) e, se richiesto, genera report opzionali con matrici di confusione (`--report-dir`) e un file JSONL con le predizioni (`--predictions`). Se `--output` non è specificato, le metriche vengono stampate su stdout in JSON, facilitando l'utilizzo in pipeline CI.
