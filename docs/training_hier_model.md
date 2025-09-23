# Training del modello gerarchico

`training/hier_trainer.py` replica la struttura del trainer a label embedding adattandola a `MultiTaskBERTMasked`. Il dataclass `HierTrainingArgs` copre iperparametri specifici come i coefficienti di ArcFace (`arcface_s`, `arcface_m`), il peso della loss di categoria (`lambda_cat`) e il `label_smoothing_super`. La pipeline prepara i dataset con `_build_dataset`, che gestisce tokenizzazione e tensori delle proprietà in modo coerente con il trainer precedente.

Anche qui è disponibile l'opzione `balanced_sampler` per contrastare sbilanciamenti: quando attivata, il dataloader di training viene ricreato con un `WeightedRandomSampler` che equilibra le classi di categoria. Dopo aver caricato tokenizer e backbone, il trainer costruisce il modello passando la maschera ontologica e l'ID `#N/D`. Il `Trainer` di Hugging Face è configurato con scheduler selezionabile e callback `SanitizeGrads` per prevenire overflow numerici.

Al termine dell'addestramento, il codice salva checkpoint completi (`pytorch_model.bin`, tokenizer) e produce un pacchetto `export/` con pesi in formato `safetensors`, label map e ontologia, replicando lo schema di deployment previsto per il label model.
