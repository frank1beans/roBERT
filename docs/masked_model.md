# Classificatore gerarchico multi-task

Il file `models/masked_model.py` implementa `MultiTaskBERTMasked`, un modello che combina classificazione gerarchica e predizione di proprietà. Analogamente al label model, utilizza un encoder Transformer condiviso, una fase di mean pooling (`MeanPool`) e una testa `EmbHead` con normalizzazione opzionale. Le predizioni sui nodi `super` e `cat` sono generate da teste ArcFace (`ArcMarginProduct`) che applicano margini angolari per incrementare la separazione tra classi; in alternativa possono essere sostituite da layer lineari standard.

Il costruttore richiede una matrice di maschera super→cat già validata, l'identificativo `#N/D`, e parametri per controllare label smoothing, normalizzazione e ritorno concatenato dei logit. Come per il label model sono previste teste ausiliarie per presenza e regressione delle proprietà, abilitate quando `num_properties > 0` e accompagnate da mask che limitano gli slot rilevanti per categoria.

Durante la forward il modello applica la maschera sulle logit di categoria, garantendo che i punteggi non validi vengano impostati a valori molto negativi (`_very_neg_like`). Questa architettura è adatta a scenari in cui si desidera ottimizzare simultaneamente la classificazione gerarchica e l'estrazione di attributi strutturati.
