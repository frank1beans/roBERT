# Comando `robimb pack`

Il comando `pack` compie l'operazione inversa di `robimb extract`: a partire da una cartella con sottostruttura `registry/` ed `extractors/` produce un bundle versionato sotto `pack/`. Oltre a `registry.json` ed `extractors.json`, il comando genera automaticamente i file vuoti `validators.json`, `formulas.json`, `views.json`, `templates.json`, `profiles.json`, `contexts.json` e `manifest.json`, aggiornando opzionalmente il symlink `pack/current/`.

Parametri principali:

- `--properties-root`: cartella che contiene le sottocartelle con gli schemi e gli estrattori;
- `--pack-root`: directory che ospita i bundle versionati (default `pack/` nel repository);
- `--version`: etichetta della nuova versione (se omessa viene calcolata automaticamente);
- `--set-current/--no-set-current`: abilita l'aggiornamento del symlink `pack/current/`.

Per compatibilità è ancora possibile specificare `--out-registry` e `--out-extractors` per ottenere i due file JSON senza creare una directory versionata. In tutti i casi il comando delega a `registry.pack_folders_to_monolith` e stampa un riepilogo con i percorsi generati, così da facilitarne l'integrazione in pipeline CI o script di pubblicazione.
