# Pack v1

## Normalizzatori supportati

I normalizzatori disponibili per questo pack sono documentati per facilitare
l'allineamento tra configurazione e runtime. Ogni voce corrisponde ad una
implementazione in `robimb.features.extractors`.

- `as_string` – restituisce il testo catturato così com'è (cast a stringa).
- `collect_many` – aggrega più match nella stessa proprietà.
- `comma_to_dot` – sostituisce le virgole decimali con punti.
- `concat_dims` – unisce due catture dimensionali nel formato `L×W`.
- `cm_to_mm?` – se il match originale è espresso in centimetri, converte il
  valore numerico in millimetri.
- `dims_join` – alias storico per `concat_dims`.
- `dot_to_comma` – sostituisce `.` con `,`.
- `format_EI_from_last_int` – produce un valore `EI {n}` usando l'ultima
  cattura numerica compatibile.
- `EI_from_any` – fallback storico per ricavare una classe EI dall'ultima
  cifra presente nel match.
- `if_cm_to_mm` – converte centimetri in millimetri quando richiesto.
- `lower` / `upper` / `strip` – trasformazioni base sulle stringhe.
- `map_enum:<nome>` – mapping tabellare definito nel pack.
- `normalize_foratura` – normalizza le descrizioni della foratura del laterizio
  (`semi pieno`, `semi-pieno`, ecc.).
- `take_last_int->EI {n}` – alias di `format_EI_from_last_int` utilizzato dai
  pacchetti legacy.
- `to_float` – converte il valore in floating point (gestendo virgole).
- `to_int` – converte il valore in intero.
- `unique_list` – rimuove duplicati mantenendo l'ordine.

L'elenco coincide con le chiavi presenti in `extractors.json`. In caso di nuove
configurazioni aggiungere qui la descrizione ed assicurarsi che esista la
relativa implementazione.
