# Script di training

I comandi legacy di training non sono più esposti direttamente dalla console `robimb`. Rimangono comunque disponibili come script dedicati basati su `argparse`:

- `python -m robimb.cli.train label ...` per il label model;
- `python -m robimb.cli.train hier ...` per il modello gerarchico multi-task;
- `python -m robimb.training.tapt_mlm ...` per l'eventuale TAPT/MLM.

L'esecuzione via `python -m ...` preserva tutti gli argomenti avanzati delle utility originali (file di configurazione, override da terminale, pubblicazione su Hugging Face Hub) mantenendo al contempo il router Typer focalizzato sui flussi operativi principali (convert, extract, pack, evaluate).
