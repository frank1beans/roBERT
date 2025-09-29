# Comando `robimb extract`

Il comando `extract` converte un knowledge pack monolitico (registry + extractors) nella struttura a cartelle utilizzata dal progetto. Gli argomenti principali sono:

- `--registry`: file JSON con il registry delle proprietà;
- `--extractors`: file JSON o pack con le regole di estrazione;
- `--out-dir`: cartella di destinazione in cui verranno ricostruite le sottocartelle `registry/` ed `extractors/`.

Il comando richiama `props.unpack.convert_monolith_to_folders` e stampa un riepilogo con la cartella generata. È utile quando si riceve un knowledge pack distribuito come singolo bundle e si vuole modificarne rapidamente la struttura o versionarlo insieme al codice.
