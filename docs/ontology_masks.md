# Maschera ontologica e report diagnostici

La maschera ontologica è una matrice S×C che impone vincoli tra classi `super` e `cat`. Viene costruita da `utils.dataset_prep.build_mask_and_report`, che a sua volta richiama `utils.ontology_utils.build_mask_from_ontology` quando è disponibile un file `ontology.json`. In assenza di ontologia, la funzione crea una matrice piena di `1.0` e registra nel report la nota *"no ontology provided"* per segnalare l'assenza di vincoli.

Quando l'ontologia è presente, il report restituito include il numero di super e categorie mancanti, oltre a statistiche di copertura utilizzate durante la conversione. Il trainer del label model (`training/label_trainer.py`) e quello gerarchico (`training/hier_trainer.py`) consumano la maschera per inizializzare `mask_matrix` e per generare maschere dinamiche in fase di inferenza (`LabelEmbedModel._build_pred_mask`). Una riga vuota nella maschera viene sanata forzando la possibilità di predire tutte le categorie, evitando errori di training.

Il report (`mask_report.json`) viene salvato dal comando `convert` insieme ai dataset preprocessati. È uno strumento diagnostico utile per individuare eventuali disallineamenti tra l'ontologia fornita e le label map generate, oltre che per monitorare l'effetto di modifiche strutturali nelle gerarchie di dominio.
