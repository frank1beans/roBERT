# Comando `robimb pack-merge`

Il comando `pack-merge` unifica knowledge pack legacy e versioni di produzione in un bundle coerente. Gli argomenti principali sono `--data-dir` (contenitore dei JSON storici), `--out-dir` (destinazione del pack versionato), `--version` (stringa semantica) e `--current-dir` (cartella in cui scrivere il symlink/log `pack.json`). Il flag `--update-current/--no-update-current` controlla se aggiornare automaticamente il pack corrente dopo la fusione.

Sotto il cofano viene richiamata la funzione `data.build_merged_pack`, che restituisce un oggetto con la lista di file generati e metadati di generazione. Se richiesto, `data.write_pack_index` produce un indice compatibile con il servizio FastAPI. Il comando stampa un riepilogo JSON con percorso dei file e manifest, utile per tracciare versioni e automatizzare deploy.
