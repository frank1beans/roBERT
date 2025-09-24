# Gestione del knowledge pack

I moduli sotto `core/` definiscono i contratti e le utility per i knowledge pack. `pack_loader.py` dichiara il dataclass `KnowledgePack` (registry, catmap, templates, validators, ecc.) e la funzione `load_pack`, che legge `pack.json`, risolve i percorsi relativi e carica in memoria i singoli componenti JSON.

Il repository distribuisce direttamente un pack pronto all'uso in `pack/current/pack.json`, generato offline a partire dalle risorse proprietarie. Gli script applicativi (CLI e servizio FastAPI) lo caricano tramite `pack_loader.load_pack`, mantenendo percorsi relativi all'interno dell'archivio. Quando è necessario aggiornare il bundle occorre rigenerare manualmente i singoli JSON (registry, estrattori, validators, viste, profili) e aggiornare l'indice `pack/current/pack.json` con i nuovi riferimenti.
