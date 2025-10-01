# ADR 0001 — Registry schema-first per l'estrazione di proprietà

- **Status**: Accepted (2024-12-01)
- **Decisori**: Tech Lead estrazione proprietà, Lead ML Engineer, Product Owner finiture
- **Stakeholder consultati**: Data engineer pack, Referente dominio finiture, Responsabile QA

## Contesto

Il bundle `pack/v1_limited` fornito dal team dominio conteneva un monolite `registry.json` privo di tipizzazione formale e con
slot ripetuti in categorie diverse. La fase A della roadmap richiede di costruire fondamenta affidabili per la pipeline ibrida,
garantendo validazione consistente, tracciabilità delle proprietà e strumenti CLI per l'ispezione degli schemi. Le alternative
valutate includevano:

1. Continuare con il monolite legacy, applicando validazione ad-hoc nel codice.
2. Migrare a un database relazionale centrale mantenendo il runtime invariato.
3. Definire un registry schema-first in JSON arricchito da JSON Schema per ogni categoria.

## Decisione

Adottiamo un registry schema-first versionato in `data/properties/registry.json` accompagnato da sette JSON Schema dedicati in
`data/properties/schema/*.json`. Ogni categoria del pack definisce ID canonico, percorso schema, proprietà con tipologia,
unità, enum, required e fonti. Le CLI (`robimb extract schemas`, `robimb extract properties`) si appoggiano a questo registry e
alle validazioni Pydantic, mentre `src/robimb/extraction/schema_registry.py` fornisce caching e interrogazione tipizzata.
Il registry è inoltre replicato nel bundle distribuito (`pack/current/registry.json`) per garantire coerenza tra codice e pack.

## Conseguenze

- **Positive**
  - Validazione preventiva con `jsonschema` e Pydantic, riducendo errori runtime.
  - Documentazione automatica delle proprietà tramite CLI e roadmap (appendice proprietà×categoria).
  - Evoluzione facilitata a `pack/v1` aggiungendo versioni e metadati senza toccare il codice.
- **Negative/Trade-off**
  - Maggior onere iniziale nel mantenere sette schemi sincronizzati con il registry.
  - Tooling aggiuntivo per generare migrazioni quando il dominio introduce nuove proprietà.
- **Mitigazioni**
  - Script `pack_folders_to_monolith` per ricostruire il monolite legacy da directory strutturate.
  - Documentazione aggiornata in `docs/ROADMAP.md` e checklist di revisione schema a ogni PR.

## Riferimenti

- Roadmap fase A con matrice proprietà×categoria (`docs/ROADMAP.md`).
- Registry consolidato (`data/properties/registry.json`).
- Loader schema-first (`src/robimb/extraction/schema_registry.py`).
