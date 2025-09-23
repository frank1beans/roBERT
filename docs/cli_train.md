# Comando `robimb train`

Il gruppo di comandi `train` in `cli/main.py` funge da proxy verso gli script di addestramento specializzati. Typer definisce due sottocomandi: `robimb train label` e `robimb train hier`. Entrambi inoltrano gli argomenti direttamente a `cli/train.py`, che a sua volta richiama le funzioni `training.label_trainer.main` o `training.hier_trainer.main` basate su `argparse`.

Questo design preserva la CLI avanzata originale (inclusi file di configurazione, override da terminale e pubblicazione su Hub) evitando duplicazioni di logica. Se non vengono forniti argomenti, ciascun sottocomando mostra l'help dettagliato generato da `argparse`, mentre eventuali errori vengono convertiti in `typer.Exit` per mantenere il comportamento coerente con gli altri comandi Typer.
