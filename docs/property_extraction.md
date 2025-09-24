# Estrazione proprietà testuali

Il pacchetto `robimb.extraction` implementa un motore di estrazione basato su pattern regex configurabili. Ogni `Pattern` associa un `property_id` a una lista di espressioni regolari e a una sequenza di normalizzatori dichiarati nel pacchetto di estrattori (`extractors_pack`). I normalizzatori built-in spaziano da conversioni numeriche (`to_float`, `to_int`) a manipolazioni di stringa (`lower`, `strip`) e trasformazioni dominio-specifiche (`EI_from_any`, `normalize_foratura`). Il modulo storico `features/extractors.py` resta come livello di compatibilità che re-esporta le nuove API.

Gli asset di default (pattern e normalizzatori) sono distribuiti nel knowledge pack `pack/current/pack.json`. La funzione `robimb.extraction.resources.load_default()` restituisce la sezione `extractors` del pack e rappresenta il riferimento usato da CLI e servizio, mentre le altre sezioni (`registry`, `catmap`, `templates`, `validators`, ecc.) vengono caricate tramite `robimb.core.pack_loader.load_pack()`.


La funzione `_compile_patterns` filtra i pattern in base all'elenco di proprietà consentite e produce oggetti immutabili efficienti per la fase di matching. Durante l'estrazione (`extract_properties`) ogni regex viene applicata al testo, i match vengono normalizzati e aggregati secondo le regole definite; l'opzione `collect_many` consente di mantenere liste di valori anziché singoli campi.

Le proprietà estratte vengono iniettate nella pipeline di conversione (`utils.data_utils.prepare_classification_dataset`) e successivamente sfruttate dai modelli durante l'addestramento multi-task per la previsione dei valori di proprietà. Questo approccio rende modulare l'aggiunta di nuovi schemi: è sufficiente aggiornare il pacchetto JSON di estrattori senza toccare il codice Python.

## Aggiornamento del pack e verifica

Quando si dispone di nuovi pattern è sufficiente aggiornare la sezione `extractors` del knowledge pack `pack/current/pack.json` (o generare un nuovo pack completo) mantenendo la stessa struttura JSON. È consigliabile eseguire uno script di validazione (ad esempio un breve check custom) che invochi `robimb.extraction.engine.validate_extractors_pack`: la funzione pre-compila le regex, verifica la presenza dei normalizzatori e restituisce errori contestualizzati, permettendo di integrare il controllo sia in CI sia in pipeline manuali.

Per estendere l'ontologia e i relativi schemi di proprietà è possibile introdurre file JSON personalizzati e passarli agli script CLI tramite `--properties-registry` e `--extractors-pack`. Entrambe le opzioni accettano anche un knowledge pack completo, dal quale verranno estratte automaticamente le sezioni di interesse. Le proprietà condividono i medesimi `property_id`, quindi l'aggiunta di nuove categorie o varianti linguistiche non richiede modifiche alla logica Python. Se servono normalizzatori personalizzati è possibile definirli nel registro built-in (`robimb.extraction.normalizers`) o tramite `map_enum:<nome>` direttamente nel pack.
