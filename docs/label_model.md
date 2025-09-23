# Modello a label embedding

`models/label_model.py` definisce `LabelEmbedModel`, un classificatore che proietta testi e label nello stesso spazio vettoriale. Il backbone è un encoder Hugging Face caricato con `AutoModel.from_pretrained`, seguito da un blocco di mean pooling (`MeanPool`) e da una testa MLP (`EmbHead`) che produce embedding normalizzati opzionalmente via L2.

Le rappresentazioni delle classi vengono inizializzate codificando i testi descrittivi (`_encode_label_texts`) e possono essere congelate o rese addestrabili (`train_label_emb`). Durante la forward vengono calcolati logit super e cat sfruttando il prodotto scalare tra embedding del documento e prototipi, scalato da un parametro `logit_scale` trainabile. Il modello supporta una `mask_matrix` derivata dall'ontologia per inibire combinazioni super→cat non valide e gestisce un ID `#N/D` con l'opzione `ban_nd_in_eval`.

Oltre alla classificazione, `LabelEmbedModel` include due teste opzionali per le proprietà: `property_presence_head` (classificazione multi-label) e `property_regression_head` (regressione), controllate da mask `property_cat_mask` e `property_numeric_mask`. Le perdite sono pesate tramite `property_presence_weight` e `property_regression_weight`, consentendo di bilanciare l'obiettivo principale con le attività ausiliarie.
