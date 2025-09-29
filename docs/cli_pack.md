# Comando `robimb pack`

Il comando `pack` compie l'operazione inversa di `robimb extract`: a partire da una cartella con sottostruttura `registry/` ed `extractors/` produce due file JSON (`--out-registry`, `--out-extractors`) pronti per essere distribuiti o archiviati come knowledge pack.

Parametri principali:

- `--properties-root`: cartella che contiene le sottocartelle con gli schemi e gli estrattori;
- `--out-registry`: percorso di destinazione del registry consolidato;
- `--out-extractors`: percorso di destinazione delle regole di estrazione consolidate.

La funzione delega a `registry.pack_folders_to_monolith` e stampa un riepilogo con i percorsi dei file generati, così da facilitarne l'integrazione in pipeline CI o script di pubblicazione.
