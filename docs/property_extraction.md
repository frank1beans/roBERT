# Estrazione proprietà testuali

Il pacchetto `robimb.extraction` raccoglie oggi i mattoni della pipeline schema-first (registry, parser deterministici, orchestrazione LLM). Il motore regex storico è stato isolato nel sotto-modulo `robimb.extraction.legacy`, che continua a esporre `Pattern`, i normalizzatori built-in e le funzioni `extract_properties`/`dry_run` per garantire piena retrocompatibilità. Ogni `Pattern` associa un `property_id` a una lista di espressioni regolari e a una sequenza di normalizzatori dichiarati nel pacchetto di estrattori (`extractors_pack`).

Gli asset di default (pattern e normalizzatori) sono distribuiti nel knowledge pack versionato (`pack/current/extractors.json`). Il motore regex può essere validato con `robimb.extraction.legacy.validate_extractors_pack`, mentre le altre sezioni (`registry`, `catmap`, `templates`, `validators`, ecc.) vengono caricate tramite `robimb.registry.load_pack()`.


La funzione `_compile_patterns` filtra i pattern in base all'elenco di proprietà consentite e produce oggetti immutabili efficienti per la fase di matching. Durante l'estrazione (`extract_properties`) ogni regex viene applicata al testo, i match vengono normalizzati e aggregati secondo le regole definite; l'opzione `collect_many` consente di mantenere liste di valori anziché singoli campi.

Le proprietà estratte vengono iniettate nella pipeline di conversione (`utils.dataset_prep.prepare_classification_dataset`) e successivamente sfruttate dai modelli durante l'addestramento multi-task per la previsione dei valori di proprietà. Questo approccio rende modulare l'aggiunta di nuovi schemi: è sufficiente aggiornare il pacchetto JSON di estrattori senza toccare il codice Python.

## Aggiornamento del pack e verifica

Quando si dispone di nuovi pattern è sufficiente generare un nuovo bundle versionato (`pack/vX/extractors.json`) mantenendo la stessa struttura JSON e aggiornare il symlink `pack/current/`. È consigliabile eseguire uno script di validazione (ad esempio un breve check custom) che invochi `robimb.extraction.legacy.validate_extractors_pack`: la funzione pre-compila le regex, verifica la presenza dei normalizzatori e restituisce errori contestualizzati, permettendo di integrare il controllo sia in CI sia in pipeline manuali.

Per estendere l'ontologia e i relativi schemi di proprietà è possibile introdurre file JSON personalizzati e passarli agli script CLI tramite `--properties-registry` e `--extractors-pack`. Entrambe le opzioni accettano anche un knowledge pack completo, dal quale verranno estratte automaticamente le sezioni di interesse. Le proprietà condividono i medesimi `property_id`, quindi l'aggiunta di nuove categorie o varianti linguistiche non richiede modifiche alla logica Python. Se servono normalizzatori personalizzati è possibile definirli nel registro built-in (`robimb.extraction.legacy.normalizers`) o tramite `map_enum:<nome>` direttamente nel pack.
