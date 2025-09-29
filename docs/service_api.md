# Servizio FastAPI

Il modulo `service/app.py` espone un'API HTTP per integrare il modello in applicazioni esterne. Variabili d'ambiente configurano percorsi di default (`ROBIMB_PACK`, `ROBIMB_MODEL`, `ROBIMB_LABEL_INDEX`, `ROBIMB_CALIBRATOR`). Per evitare ricarichi ripetuti, il file mantiene uno stato condiviso `_state` protetto da lock (`_pack_lock`, `_model_lock`). Le funzioni `_load_pack_once` e `_load_model_once` caricano rispettivamente knowledge pack e modello solo quando cambia il percorso richiesto.

L'applicazione definisce due endpoint principali: `GET /health`, che restituisce stato e percorsi configurati, e `POST /predict`, che riceve un payload `PredictIn` (testo, top-k desiderato, contesto opzionale, override percorsi). Il body viene validato con Pydantic e passato alle routine di inferenza: `predict_topk` per la classificazione, `predict_properties` per l'estrazione, `robimb.registry.validate` per le regole e `templates.render` per la descrizione. L'output `PredictOut` riepiloga categoria, top-k, proprietà, issue e descrizione.

Questo servizio riutilizza gli stessi helper della CLI garantendo coerenza dei risultati e permette override runtime (es. puntare a un knowledge pack temporaneo) senza riavviare il processo.
