# Gestione del knowledge pack

Il package `robimb.registry` definisce i contratti e le utility per i knowledge pack. `loader.py` dichiara la dataclass `RegistryBundle` (versione, registry, catmap, templates, validators, ecc.) e la funzione `load_pack`, che legge `pack.json` e restituisce tutte le sezioni del bundle insieme ai modelli Pydantic per categorie e proprietà.

Il repository distribuisce direttamente un pack pronto all'uso in `pack/current/pack.json`, generato offline a partire dalle risorse proprietarie e già in forma "inline" (tutti i componenti sono contenuti nello stesso file). Gli script applicativi (CLI e servizio FastAPI) lo caricano tramite `robimb.registry.load_pack` oppure consumano singole sezioni (`extractors`, `templates`, ...) leggendo il medesimo JSON. Per aggiornare il bundle è sufficiente rigenerare le sezioni necessarie e sostituire il file, senza doversi occupare di riferimenti incrociati a percorsi esterni.

