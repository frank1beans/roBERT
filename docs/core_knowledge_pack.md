# Gestione del knowledge pack

I moduli sotto `core/` definiscono i contratti e le utility per i knowledge pack. `pack_loader.py` dichiara il dataclass `KnowledgePack` (registry, catmap, templates, validators, ecc.) e la funzione `load_pack`, che legge `pack.json`, risolve i percorsi relativi e carica in memoria i singoli componenti JSON.

Le routine operative per unire gli asset legacy con quelli di produzione sono ora concentrate in `robimb.data.pack_merge`. Il modulo espone `build_merged_pack`, che compone registry, estrattori, validatori e viste in un bundle coerente, produce il `manifest.json` con digest e metadati e aggiorna l'indice nella cartella `pack/current/`. Le stesse funzioni sono utilizzate dal comando `robimb pack-merge` e dagli script di conversione per distribuire automaticamente i nuovi pack.
