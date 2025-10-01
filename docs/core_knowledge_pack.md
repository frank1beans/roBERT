# Gestione del knowledge pack

Il package `robimb.registry` definisce i contratti e le utility per i knowledge pack. `loader.py` dichiara la dataclass `RegistryBundle` (versione, registry, catmap, templates, validators, ecc.) e la funzione `load_pack`, che legge una directory o un file `pack.json` e restituisce tutte le sezioni del bundle insieme ai modelli Pydantic per categorie e proprietà.

Il repository distribuisce direttamente due varianti del knowledge pack:

- `resources/pack/v1_limited/` (impostato come target del symlink `resources/pack/current/`) contiene una selezione di categorie focalizzate sulle finiture: opere da cartongessista, controsoffitti, pavimentazioni, rivestimenti, opere da falegname, opere da serramentista e apparecchi sanitari. Ogni categoria espone soltanto gli slot essenziali – marchio, produttore e le proprietà specifiche richieste (spessori, materiali, prestazioni EI/Rw, dimensioni, ecc.) – insieme ai relativi pattern di estrazione e validatori.
- `resources/pack/v1/` resta disponibile come bundle completo per casi d'uso avanzati o per chi necessita dell'intero catalogo storico.

Entrambe le versioni sono suddivise in file distinti (`registry.json`, `extractors.json`, `validators.json`, `formulas.json`, `views.json`, `templates.json`, `profiles.json`, `contexts.json` e `manifest.json`). Gli script applicativi (CLI e servizio FastAPI) caricano la directory tramite `robimb.registry.load_pack` oppure consumano singole sezioni, ad esempio `extractors.json` validato con `robimb.extraction.legacy.validate_extractors_pack`. Per aggiornare il bundle è sufficiente creare una nuova directory versione (`resources/pack/v2/…`) con la stessa struttura e aggiornare il symlink `resources/pack/current/`.

