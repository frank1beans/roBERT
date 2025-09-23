# Estrazione proprietà testuali

Il pacchetto `robimb.extraction` implementa un motore di estrazione basato su pattern regex configurabili. Ogni `Pattern` associa un `property_id` a una lista di espressioni regolari e a una sequenza di normalizzatori dichiarati nel pacchetto di estrattori (`extractors_pack`). I normalizzatori built-in spaziano da conversioni numeriche (`to_float`, `to_int`) a manipolazioni di stringa (`lower`, `strip`) e trasformazioni dominio-specifiche (`EI_from_any`, `normalize_foratura`). Il modulo storico `features/extractors.py` resta come livello di compatibilità che re-esporta le nuove API.

Gli asset di default (pattern e normalizzatori) sono versionati in `src/robimb/extraction/resources/`. Il file principale `extractors.json` può essere caricato direttamente via `robimb.extraction.resources.load_default()` e rappresenta il riferimento usato dalla CLI e dall'inferenza. Lo stesso percorso ospita anche la base legacy `extractors_patterns.json`, utile per mantenere la storia delle regole prima del merge con la controparte produttiva.

La funzione `_compile_patterns` filtra i pattern in base all'elenco di proprietà consentite e produce oggetti immutabili efficienti per la fase di matching. Durante l'estrazione (`extract_properties`) ogni regex viene applicata al testo, i match vengono normalizzati e aggregati secondo le regole definite; l'opzione `collect_many` consente di mantenere liste di valori anziché singoli campi.

Le proprietà estratte vengono iniettate nella pipeline di conversione (`utils.data_utils.prepare_classification_dataset`) e successivamente sfruttate dai modelli durante l'addestramento multi-task per la previsione dei valori di proprietà. Questo approccio rende modulare l'aggiunta di nuovi schemi: è sufficiente aggiornare il pacchetto JSON di estrattori senza toccare il codice Python.

## Aggiornamento del pack e verifica

Il file `scripts/build_extractors_pack.py` genera automaticamente le risorse `extractors*.json`, il pack corrente (`pack/current/pack.json`) e la checklist `docs/extraction_checklist.md` a partire da `data/properties_registry_extended.json`. Il comando:

```bash
python scripts/build_extractors_pack.py
```

aggiorna tutte le risorse mantenendo allineati pattern, normalizzatori e metadati (`language`, `confidence`, `tags`). Il generatore applica anche varianti regex per le principali unità (es. `mq`, `m²`, `metri quadri`) e aggiunge normalizzatori built-in per unità, liste strutturate e mapping multilingua.

Ogni pack può essere validato sintatticamente senza eseguire l'estrazione grazie alla funzione `validate_extractors_pack` del modulo `robimb.extraction.engine`, che pre-compila tutte le regex e restituisce errori contestualizzati. Questo permette di integrare il controllo sia in CI sia in pipeline manuali prima di distribuire una nuova versione del pack.

Per estendere l'ontologia è sufficiente intervenire su `properties_registry_extended.json` (aggiungendo slot, priorità e pattern) e rigenerare gli asset. Le proprietà condividono i medesimi `property_id`, quindi l'aggiunta di nuove categorie o varianti linguistiche non richiede modifiche alla logica Python. Se servono normalizzatori personalizzati è possibile definirli nel registro built-in (`robimb.extraction.normalizers`) o tramite `map_enum:<nome>` direttamente nel pack.
