# Gestione del knowledge pack

I moduli sotto `core/` definiscono i contratti e le utility per i knowledge pack. `pack_loader.py` dichiara il dataclass `KnowledgePack` (registry, catmap, templates, validators, ecc.) e la funzione `load_pack`, che legge `pack.json`, risolve i percorsi relativi e carica in memoria i singoli componenti JSON.

`pack_tools.py` fornisce funzioni di supporto per costruire manifest e aggiornare il pack corrente. `build_manifest` copia i file chiave in una directory di destinazione, calcola hash SHA-256 e scrive `manifest.json` con metadati (dimensioni, digest, timestamp). `update_current` genera `pack/current/pack.json` puntando alla versione appena creata tramite percorsi relativi.

Infine `pack_merge.py` e `pack_validate.py` (non trattati in dettaglio qui) orchestrano rispettivamente la fusione di pack legacy e la validazione degli schemi. Queste routine sono richiamate dal comando `robimb pack-merge` e assicurano che il servizio e la CLI lavorino sempre con un bundle coerente e tracciabile.
