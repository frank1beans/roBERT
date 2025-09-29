# Reportistica e analisi

La cartella `reporting/` contiene utility per generare grafici e riepiloghi durante la conversione dei dataset e la valutazione dei modelli. `dataset_reports.py` produce statistiche descrittive (lunghezza media dei testi, distribuzioni di super/cat) e salva grafici PNG utilizzando Matplotlib/Seaborn in modalità headless (`matplotlib.use("Agg")`). I risultati vengono consolidati in `dataset_summary.json` e referenziati dal comando `convert` quando viene specificata l'opzione `--reports-dir`.

`prediction_reports.py` si concentra invece sulle valutazioni: calcola confusion matrix normalizzate per le classi più frequenti, genera report testuali (`classification_report`) e identifica le coppie di classi più confuse. Anche in questo caso gli output sono grafici PNG e file JSON, pensati per essere consultati in fase di validazione (`robimb evaluate`).

Questi strumenti aiutano a diagnosticare sbilanciamenti, individuare pattern di errore e documentare le performance dei modelli lungo il ciclo di vita.
