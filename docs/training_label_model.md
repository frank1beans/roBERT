# Training del label model

`training/label_trainer.py` definisce `LabelTrainingArgs`, un dataclass che raccoglie tutte le opzioni CLI (checkpoint di partenza, file di training/validation, pesi di loss, pubblicazione su Hub). La funzione `train_label_model` gestisce il flusso: inizializza gli seed, crea la directory di output e carica le label map per determinare `num_super` e `num_cat`. Successivamente genera la maschera ontologica tramite `build_mask_and_report` e carica tokenizer e backbone indicati dagli argomenti.

I dataset vengono costruiti con `_build_dataset`, che tokenizza i testi, allega gli indici numerici e prepara i tensori delle proprietà usando `build_property_targets`. Se `balanced_sampler` è attivo viene impiegato un `WeightedRandomSampler` basato sulla frequenza delle categorie. L'oggetto `LabelEmbedModel` può essere inizializzato da un checkpoint esistente (`init_from`) e supporta l'unfreezing progressivo degli ultimi layer tramite `unfreeze_last`.

L'addestramento vero e proprio utilizza `transformers.Trainer` con `TrainingArguments` configurati dinamicamente (scheduler, warmup, gradient accumulation). Il callback `SanitizeGrads` monitora i gradienti NaN/Inf e li ripristina a zero per evitare instabilità. Al termine, il trainer salva i migliori pesi, esporta i prototipi delle label e, opzionalmente, pubblica l'artefatto sul modello indicato (`hub_repo`).
