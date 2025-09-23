# Pre-training TAPT/MLM

Lo script `training/tapt_mlm.py` fornisce un'interfaccia avanzata per il Domain-Adaptive Pre-Training (TAPT) con obiettivi di Masked Language Modeling. Sono supportate opzioni orientate alle GPU moderne: Whole-Word Masking (`--wwm`), Layer-wise Learning Rate Decay (`--llrd`), congelamento dei layer inferiori (`--freeze_layers`) e sblocco progressivo (`--unfreeze_at_epoch`). Il codice abilita automaticamente TF32 su GPU Ampere/Ada e consente l'uso di BF16, gradient checkpointing e resume robusto da checkpoint.

Il flusso carica un dataset testuale tramite `datasets.load_dataset`, applica filtri di pulizia e deduplicazione (`load_text_files`), quindi segmenta il corpus in blocchi della dimensione desiderata con `prepare_mlm_blocks`. Tokenizer e modello (`AutoTokenizer`, `AutoModelForMaskedLM`) vengono istanziati in base ai parametri CLI, con supporto per collator specializzati (`DataCollatorForWholeWordMask`).

Per l'ottimizzazione viene utilizzato `Trainer` di Hugging Face, arricchito da `EarlyStoppingCallback` e da utility personalizzate come `make_llrd_param_groups` per configurare AdamW con gruppi di parametri decrescenti. Il comando `robimb tapt` è un thin wrapper che passa gli argomenti Typer direttamente a questo script, garantendo uniformità con il resto della suite.
