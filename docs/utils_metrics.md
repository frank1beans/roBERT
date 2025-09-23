# Utility per le metriche

`utils/metrics_utils.py` espone `make_compute_metrics`, factory che restituisce una funzione compatibile con `transformers.Trainer`. La closure riceve le dimensioni di super e cat e opzionalmente metadati sulle proprietà. Durante la valutazione, la funzione gestisce sia il formato tuple (logit separati) sia matrici singole e converte i tensori in NumPy gestendo eventuali NaN/Inf.

Le metriche calcolate includono accuracy e macro-F1 per le classi `super` e `cat`, distinguendo tra logit condizionati sulla predizione del super e logit condizionati sul super gold. Quando sono presenti teste di proprietà vengono calcolate anche accuracy/F1 per la presenza e MAE/RMSE per la regressione, utilizzando le maschere fornite nel batch (`property_slot_mask`, `property_regression_mask`). Questo approccio garantisce compatibilità con entrambe le architetture del progetto e fornisce segnali chiave per il monitoraggio delle performance multi-task.
