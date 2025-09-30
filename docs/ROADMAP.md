# Sintesi esecutiva
- Completato l'audit del bundle `pack/v1_limited`: identificate sette categorie operative (cartongesso, rivestimenti, pavimentazioni, serramenti, controsoffitti, apparecchi sanitari, falegnameria) con slot condivisi e prevalenza di pattern regex focalizzati su marchi e pochi valori dimensionali.
- Definito un registro schema-first (`data/properties/registry.json`) allineato alle categorie legacy con metadati strutturati (tipi, unità, enum, required, fonti) e sette JSON Schema validabili in `data/properties/schema/` per garantire tracciabilità e validazione.
- Istituito un piano incrementale A→D che combina parser deterministici, QA LLM vincolato, matcher lessicali e validazione Pydantic per evolvere dal pack limitato all'estrattore ibrido pronto per il bundle `pack/v1`.

## Assunzioni
- Dataset interno disponibile con almeno 80 descrizioni etichettate per ciascuna delle sette categorie di `pack/v1_limited`.
- Accesso a un LLM compatto (GPT-4o-mini o equivalente) con supporto JSON-constrained e latenza <1s.
- Glossari di brand, materiali, trattamenti e standard forniti dagli esperti dominio entro la fine della Fase A.
- Stack tecnologico aggiornabile a Python ≥3.11 con disponibilità di pytest, jsonschema, Typer, Pydantic v2, ruff e black in CI.
- Possibilità di generare dati sintetici controllati per coprire gap nelle classi prestazionali (PEI, Rw, EI, ecc.).

## Roadmap
### Fase A — Foundations (settimane 1-3)
- **Obiettivi**
  - Inventariare `pack/v1_limited` e documentare mapping slot→proprietà.
  - Produrre registry e JSON Schema per le sette categorie legacy.
  - Implementare parser deterministici numeri/unità/dimensioni e skeleton CLI Typer.
- **Attività principali**
  - Audit pack, ADR schema-first, definizione `data/properties/registry.json`.
  - Creazione JSON Schema in `data/properties/schema/*.json` e caricatore `schema_registry.py`.
  - Parser deterministici (`numbers.py`, `units.py`, `dimensions.py`) con suite pytest ≥60 casi.
  - Nuovo comando `robimb extract` con sottocomandi `properties`, `schemas`, `pack` e logging strutturato.
- **Dipendenze**: accesso completo a `pack/v1_limited`, conferma dei glossari iniziali, disponibilità esperto dominio per review schemi.
- **Rischi/Mitigazioni**
  - Schemi incompleti → workshop con dominio + ADR vincolanti.
  - Parser dimensioni incompleti → raccolta esempi reali, generazione casi sintetici.
- **Deliverable/DoD**
  - Registry + sette JSON Schema validati con `jsonschema` e caricabili via `schema_registry.py`.
  - Parser con pytest verde (coverage logica ≥90%) e CLI `robimb extract properties --dry-run` funzionante.

### Fase B — Pipeline ibrida (settimane 4-7)
- **Obiettivi**
  - Introdurre QA estrattivo JSON-constrained, matchers lessicali e fusione candidati.
  - Implementare orchestrator, normalizzatori, validazione Pydantic e logging span-level.
- **Attività principali**
  - `prompts.py`, `qa_llm.py`, `normalize.py`, lessici in `data/properties/lexicon/`.
  - Matchers brand/materiali/standard/colore e parser aggiuntivi (colori RAL, norme UNI/EN).
  - `fuse.py`, `validators.py`, `orchestrator.py` con test integrazione su batch multi-categoria.
- **Dipendenze**: completamento Fase A, API LLM disponibile, lexicon iniziali.
- **Rischi/Mitigazioni**
  - Risposte LLM non conformi → validazione schema, retry, fallback parser.
  - Lessici incompleti → pipeline aggiornamento da CSV + fallback QA con soglie.
- **Deliverable/DoD**
  - QA extractor con EM ≥0.75 su golden set multi-categoria.
  - Fusione con output JSONL (value/unit/source/span/confidence) e CLI batch end-to-end.

### Fase C — Qualità & Metriche (settimane 8-10)
- **Obiettivi**
  - Generare dataset validazione (≥350 record) bilanciato sulle sette categorie.
  - Implementare metriche unit-aware, report automatici, soglie di confidenza e CI dedicata.
- **Attività principali**
  - `data/properties/validation/pilota.jsonl`, `src/robimb/utils/metrics.py`, reporting Markdown/HTML.
  - CLI `robimb extract evaluate` con generazione report, soglie, grafici coverage.
- **Dipendenze**: pipeline ibrida stabile, dataset etichettato.
- **Rischi/Mitigazioni**
  - Dataset sbilanciato → synthetic data + stratified sampling.
  - Metriche lente → vectorization (numpy/pandas) e caching intermediate.
- **Deliverable/DoD**
  - Report KPI con coverage ≥0.8 e EM macro ≥0.7, generato da CLI.
  - Workflow CI che esegue eval e pubblica badge.

### Fase D — Hardening (settimane 11-14)
- **Obiettivi**
  - Gestire edge case (range, ≥, simboli), normalizzazione avanzata, performance e packaging `pack/v1`.
  - Documentazione finale, packaging PyPI e knowledge transfer.
- **Attività principali**
  - Estensione parser range/comparatori, normalizzatori RAL/UNI, caching orchestrator.
  - Benchmark throughput (`tests/perf/test_throughput.py`), packaging wheel, documentazione aggiornata.
- **Dipendenze**: metriche consolidate Fase C, dataset completo.
- **Rischi/Mitigazioni**
  - Performance < target → parallel Typer + batching LLM + caching lexicon.
  - Divergenza pack/codice → CI con checksum e validazione pack.
- **Deliverable/DoD**
  - `pack/v1/` completo con manifest, registry, prompts, lexicon aggiornati.
  - Benchmark ≤45s/1000 record e documentazione aggiornata (`docs/cli_extract.md`, `docs/property_extraction.md`).

## Backlog
- **A1. Audit pack v1_limited** — File: `pack/v1_limited/*`, `docs/ROADMAP.md`. DoD: tabella slot→categoria in Appendice; review TL. Stima: 6h.
- **A2. ADR schema-first** — File: `docs/ADR/0001-schema-first.md`. DoD: decisione approvata con pro/contro, linkata dalla roadmap. Stima: 5h.
- **A3. Registry completo** — File: `data/properties/registry.json`. DoD: validazione jsonschema, 7 categorie con metadata completi. Stima: 8h.
- **A4. JSON Schema categorie** — File: `data/properties/schema/*.json`. DoD: validazione `jsonschema` + smoke test caricamento. Stima: 12h.
- **A5. Schema registry loader** — File: `src/robimb/extraction/schema_registry.py`. DoD: API `load_registry` con cache + test `tests/test_schema_registry.py`. Stima: 8h.
- **A6. Validators Pydantic** — File: `src/robimb/extraction/validators.py`. DoD: `validate_properties` restituisce errori strutturati; coverage ≥90%. Stima: 9h.
- **A7. Parser numerici** — File: `src/robimb/extraction/parsers/numbers.py`. DoD: funzioni `parse_number_it`/`extract_numbers`, pytest ≥50 casi. Stima: 8h.
- **A8. Parser dimensioni** — File: `src/robimb/extraction/parsers/dimensions.py`. DoD: supporto formati `90x210`, `0,90×2,10`, `L90 H210`; ≥60 casi, EM ≥0.9. Stima: 12h.
- **A9. Parser unità** — File: `src/robimb/extraction/parsers/units.py`. DoD: normalizzazione mm/cm/m, m², kN/m²; pytest ≥40 casi. Stima: 7h.
- **A10. CLI skeleton** — File: `src/robimb/cli/extract.py`, `src/robimb/cli/main.py`. DoD: `robimb extract properties --help` e `--dry-run` attivi, snapshot help. Stima: 6h.
- **A11. Logging strutturato base** — File: `src/robimb/utils/logging.py`. DoD: log JSONL con trace-id testato via snapshot. Stima: 5h.
- **A12. Coverage tests schema** — File: `tests/test_schema_registry.py`. DoD: assert categorie/required match registry. Stima: 4h.
- **B1. Prompt library** — File: `src/robimb/extraction/prompts.py`, `data/properties/prompts.json`. DoD: template LLM con test rendering. Stima: 7h.
- **B2. QA LLM adapter** — File: `src/robimb/extraction/qa_llm.py`. DoD: retry/backoff + mock tests error handling. Stima: 10h.
- **B3. Lexicon brand/materiali** — File: `data/properties/lexicon/brands.txt`, `materials.txt`. DoD: ≥250 voci, script dedup. Stima: 6h.
- **B4. Matcher marchi** — File: `src/robimb/extraction/matchers/brands.py`. DoD: precision ≥0.9 su 200 esempi. Stima: 8h.
- **B5. Matcher materiali** — File: `src/robimb/extraction/matchers/materials.py`. DoD: sinonimi/lemmatizzazione, EM ≥0.8. Stima: 8h.
- **B6. Parser colori RAL** — File: `src/robimb/extraction/parsers/colors.py`, `data/properties/lexicon/colors_ral.json`. DoD: mapping completo, test 30 codici. Stima: 7h.
- **B7. Parser standard UNI/EN** — File: `src/robimb/extraction/parsers/standards.py`, `data/properties/lexicon/standards_prefixes.json`. DoD: recall ≥0.95 su 50 esempi. Stima: 9h.
- **B8. Normalizzatori proprietà** — File: `src/robimb/extraction/normalize.py`. DoD: funzioni per formati, classi prestazionali, booleani. Stima: 7h.
- **B9. Fusione candidati** — File: `src/robimb/extraction/fuse.py`. DoD: policy configurabile + test conflitti. Stima: 12h.
- **B10. Orchestrator pipeline** — File: `src/robimb/extraction/orchestrator.py`. DoD: pipeline completa multi-categoria, integrazione CLI. Stima: 10h.
- **B11. Logging span-level** — File: `src/robimb/extraction/fuse.py`, `orchestrator.py`. DoD: output include `source/span/confidence`, snapshot 10 record. Stima: 6h.
- **B12. Pack prompts/manifest** — File: `pack/v1/*`. DoD: manifest validato via jsonschema. Stima: 5h.
- **C1. Dataset validazione** — File: `data/properties/validation/pilota.jsonl`. DoD: ≥350 record, bilanciamento ±10% categoria. Stima: 12h.
- **C2. Metriche unit-aware** — File: `src/robimb/utils/metrics.py`. DoD: `exact_match`, `unit_aware_mae`, test numerici. Stima: 9h.
- **C3. Report qualità** — File: `src/robimb/reporting/properties_report.py`, `docs/property_extraction_report_template.md`. DoD: report Markdown/HTML rigenerabile. Stima: 10h.
- **C4. CLI evaluate** — File: `src/robimb/cli/extract.py`. DoD: sottocomando `evaluate` con test CLI. Stima: 7h.
- **C5. Confidence calibration** — File: `src/robimb/extraction/fuse.py`, `data/properties/calibration.json`. DoD: curva precision-recall, soglie per categoria. Stima: 8h.
- **C6. Coverage analytics** — File: `src/robimb/reporting/coverage.py`. DoD: grafico coverage property/category. Stima: 6h.
- **C7. CI metrics** — File: `.github/workflows/qa_properties.yml`. DoD: workflow che esegue eval e pubblica badge. Stima: 8h.
- **C8. Alerting QA** — File: `docs/ADR/0003-quality-guardrails.md`. DoD: ADR su soglie e alert. Stima: 5h.
- **D1. Parser range/comparatori** — File: `src/robimb/extraction/parsers/dimensions.py`. DoD: supporto `≥`, `da...a`, `±`, test 60 esempi. Stima: 10h.
- **D2. Normalizzatori avanzati** — File: `src/robimb/extraction/normalize.py`. DoD: `normalize_dimension_range`, `normalize_standard`, test 40 casi. Stima: 8h.
- **D3. Validator messaggi azionabili** — File: `src/robimb/extraction/validators.py`. DoD: errori con codici/suggerimenti, snapshot. Stima: 6h.
- **D4. Performance benchmark** — File: `tests/perf/test_throughput.py`. DoD: ≤45s/1000 record, profilo allegato. Stima: 9h.
- **D5. Packaging** — File: `pyproject.toml`, `src/robimb/__init__.py`. DoD: wheel installabile, CLI disponibile. Stima: 6h.
- **D6. Documentazione finale** — File: `docs/cli_extract.md`, `docs/property_extraction.md`, `docs/ROADMAP.md`. DoD: guide aggiornate, review tech writer. Stima: 7h.
- **D7. Pack v1** — File: `pack/v1/*`. DoD: registry, schema, prompts, lexicon aggiornati con checksum. Stima: 8h.
- **D8. Knowledge transfer** — File: `docs/KT/property_extraction_slides.pdf`. DoD: sessione registrata, feedback ≥4/5. Stima: 5h.

## Matrice Proprietà×Categoria
### Opere da cartongessista
| Property ID | Tipo | Unità | Regole di validazione | Esempio testo → output |
|-------------|------|-------|-----------------------|------------------------|
| `marchio` | string | — | 2–80 char | “Sistema cartongesso Knauf” → “Knauf” |
| `tipologia_lastra` | enum | — | {standard,idrofuga,ignifuga,acustica,fibrogesso,accoppiata_isolante} | “lastra idrofuga” → “idrofuga” |
| `spessore_mm` | float | mm | 6 ≤ val ≤ 100 | “parete da 12,5 mm” → 12.5 |
| `classe_ei` | enum | — | {EI30, EI45, EI60, EI90, EI120, EI180} | “parete EI 60” → “EI60” |
| `classe_reazione_al_fuoco` | enum | — | {A1, A2-s1,d0, A2-s2,d0, B-s1,d0, B-s2,d0} | “classe A2-s1,d0” |
| `presenza_isolante` | enum | — | {si,no} | “con isolante in lana minerale” → “si” |
| `stratigrafia_lastre` | string | — | 5–400 char | “doppia lastra 12,5 + isolante 50 mm” |

**Esempio testo**: “Controparete cartongesso Knauf doppia lastra idrofuga, spessore complessivo 125 mm, classe EI 60, reazione al fuoco A2-s1,d0, con isolante lana minerale.”

### Opere di rivestimento
| Property ID | Tipo | Unità | Regole di validazione | Esempio |
|-------------|------|-------|-----------------------|--------|
| `marchio` | string | — | 2–80 char | “Rivestimento Marazzi” → “Marazzi” |
| `materiale` | enum | — | {gres,pietra,ceramica,legno,resina,intonaco,laminato,metallo} | “pannelli in gres” |
| `finitura` | string | — | 3–80 char | “finitura effetto cemento” |
| `spessore_mm` | float | mm | 2 ≤ val ≤ 80 | “spessore 10 mm” |
| `posa` | enum | — | {incollata,flottante,a secco,meccanica,su struttura} | “posa a secco” |
| `classe_reazione_al_fuoco` | enum | — | {A1,A2-s1,d0,B-s1,d0,C-s1,d0} | “classe B-s1,d0” |

**Esempio testo**: “Rivestimento pareti in gres porcellanato Marazzi spessore 10 mm, posa a secco su sottostruttura, finitura effetto cemento, reazione al fuoco B-s1,d0.”

### Opere di pavimentazione
| Property ID | Tipo | Unità | Regole di validazione | Esempio |
|-------------|------|-------|-----------------------|--------|
| `marchio` | string | — | 2–80 char | “Pavimento Florim” → “Florim” |
| `materiale` | enum | — | {gres,legno,laminato,resina,pietra_naturale,calcestruzzo,vinilico} | “doghe in legno” |
| `formato` | string | — | pattern \d{2,4}x\d{2,4} con unità opzionale | “formato 600x600 mm” |
| `spessore_mm` | float | mm | 4 ≤ val ≤ 50 | “spessore 12 mm” |
| `classe_resistenza_usura` | enum | — | {PEI I, PEI II, PEI III, PEI IV, PEI V} | “classe PEI IV” |
| `classe_scivolosita` | enum | — | {R9,R10,R11,R12,R13} | “classe R11” |

**Esempio testo**: “Pavimento in gres Florim formato 600x600 mm spessore 12 mm, classe PEI IV, antiscivolo R11.”

### Opere da serramentista
| Property ID | Tipo | Unità | Regole di validazione | Esempio |
|-------------|------|-------|-----------------------|--------|
| `marchio` | string | — | 2–80 char | “Sistema Schüco” → “Schüco” |
| `materiale_struttura` | enum | — | {alluminio,acciaio,legno,pvc,legno_alluminio} | “profilo in alluminio” |
| `dimensione_larghezza` | float | mm | 400 ≤ val ≤ 4000 | “luce 1200 mm” |
| `dimensione_altezza` | float | mm | 500 ≤ val ≤ 3500 | “altezza 2400 mm” |
| `trasmittanza_termica` | float | W/m²K | 0.5 ≤ val ≤ 3.5 | “Uw 1,3 W/m²K” |
| `isolamento_acustico_db` | float | dB | 20 ≤ val ≤ 55 | “Rw 42 dB” |

**Esempio testo**: “Infisso Schüco in alluminio, luce 1200x2400 mm, Uw 1,3 W/m²K, isolamento acustico Rw 42 dB.”

### Controsoffitti
| Property ID | Tipo | Unità | Regole di validazione | Esempio |
|-------------|------|-------|-----------------------|--------|
| `marchio` | string | — | 2–80 char | “Controsoffitto Armstrong” |
| `materiale` | enum | — | {acciaio,alluminio,cartongesso,lana_minerale,fibra_minerale} | “pannelli in lana minerale” |
| `spessore_pannello_mm` | float | mm | 4 ≤ val ≤ 50 | “spessore 15 mm” |
| `classe_reazione_al_fuoco` | enum | — | {A1,A2-s1,d0,B-s1,d0,B-s2,d0} | “classe A2-s1,d0” |
| `coefficiente_fonoassorbimento` | float | — | 0.0 ≤ val ≤ 1.0 | “αw 0,85” |
| `stratigrafia_lastre` | string | — | 5–400 char | “doppio strato gesso + feltro acustico” |

**Esempio testo**: “Controsoffitto Armstrong in lana minerale, pannelli spessore 15 mm, αw 0,85, classe A2-s1,d0, doppio strato con velo acustico.”

### Apparecchi sanitari e accessori
| Property ID | Tipo | Unità | Regole di validazione | Esempio |
|-------------|------|-------|-----------------------|--------|
| `marchio` | string | — | 2–80 char | “Lavabo Ideal Standard” |
| `materiale` | enum | — | {ceramica,acciaio_inox,resina,vetro,ghisa,porcellana} | “lavabo in ceramica” |
| `dimensione_lunghezza` | float | mm | 150 ≤ val ≤ 2500 | “lunghezza 900 mm” |
| `dimensione_larghezza` | float | mm | 150 ≤ val ≤ 1500 | “larghezza 450 mm” |
| `dimensione_altezza` | float | mm | 100 ≤ val ≤ 2000 | “altezza 200 mm” |
| `tipologia_installazione` | enum | — | {a_pavimento,a_parete,sospesa,incasso} | “installazione sospesa” |
| `portata_l_min` | float | l/min | 0 ≤ val ≤ 60 | “portata 6 l/min” |

**Esempio testo**: “Lavabo sospeso Ideal Standard in ceramica 900x450x200 mm, portata rubinetto 6 l/min.”

### Opere da falegname
| Property ID | Tipo | Unità | Regole di validazione | Esempio |
|-------------|------|-------|-----------------------|--------|
| `marchio` | string | — | 2–80 char | “Porta Garofoli” |
| `essenza` | enum | — | {rovere,faggio,pino,abete,larice,noce,castagno} | “essenza rovere” |
| `dimensione_larghezza` | float | mm | 300 ≤ val ≤ 3000 | “larghezza 900 mm” |
| `dimensione_altezza` | float | mm | 400 ≤ val ≤ 3200 | “altezza 2100 mm” |
| `tipologia_apertura` | enum | — | {battente,scorrevole,a_ribalta,a_scomparsa,anta_ribalta} | “porta scorrevole” |

**Esempio testo**: “Porta interna Garofoli in rovere, dimensioni 900x2100 mm, apertura scorrevole.”

## Contratti I/O
- **Input JSONL (`data/properties/input_schema.json`)**
  ```json
  {
    "type": "object",
    "required": ["text_id", "categoria", "text"],
    "properties": {
      "text_id": {"type": "string"},
      "categoria": {"type": "string"},
      "text": {"type": "string"},
      "metadata": {"type": "object"}
    },
    "additionalProperties": false
  }
  ```
- **Output JSONL (`data/properties/output_schema.json`)**
  ```json
  {
    "type": "object",
    "required": ["text_id", "categoria", "properties"],
    "properties": {
      "text_id": {"type": "string"},
      "categoria": {"type": "string"},
      "properties": {
        "type": "object",
        "additionalProperties": {
          "type": "object",
          "required": ["value", "source", "span", "confidence"],
          "properties": {
            "value": {},
            "unit": {"type": ["string", "null"]},
            "source": {"enum": ["parser", "matcher", "qa_llm", "fuse"]},
            "raw": {"type": ["string", "null"]},
            "span": {
              "type": "array",
              "items": {"type": "integer"},
              "minItems": 2,
              "maxItems": 2
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": ["string", "null"]},
            "validation": {"type": "object"}
          }
        }
      },
      "warnings": {"type": "array", "items": {"type": "string"}},
      "errors": {"type": "array", "items": {"type": "string"}}
    },
    "additionalProperties": false
  }
  ```
- **Esempi JSONL**
  ```jsonl
  {"text_id":"cg_001","categoria":"Opere da cartongessista","text":"Controparete Knauf EI 60 spessore 125 mm con isolante lana minerale, classe A2-s1,d0."}
  {"text_id":"riv_010","categoria":"Opere di rivestimento","text":"Rivestimento Marazzi in gres 600x1200 mm spessore 10 mm, posa a secco, reazione al fuoco B-s1,d0."}
  {"text_id":"pav_025","categoria":"Opere di pavimentazione","text":"Pavimento Florim formato 600x600 mm, spessore 12 mm, classe PEI IV, antiscivolo R11."}
  {"text_id":"ser_012","categoria":"Opere da serramentista","text":"Infisso Schüco in alluminio 1200x2400 mm, Uw 1,3 W/m²K, Rw 42 dB."}
  {"text_id":"ctr_005","categoria":"Controsoffitti","text":"Controsoffitto Armstrong in lana minerale spessore 15 mm, αw 0,85, classe A2-s1,d0."}
  {"text_id":"san_018","categoria":"Apparecchi sanitari e accessori","text":"Lavabo sospeso Ideal Standard in ceramica 900x450x200 mm, portata 6 l/min."}
  {"text_id":"fal_003","categoria":"Opere da falegname","text":"Porta interna Garofoli in rovere 900x2100 mm apertura scorrevole."}
  ```

## CLI Spec
- **Comando principale**: `robimb extract`
- **Sottocomandi**
  - `robimb extract properties --input INPUT.jsonl --output OUTPUT.jsonl --pack pack/v1 --schema data/properties/registry.json --batch-size 32 --max-workers 4 --confidence-threshold 0.6 --category-filter "Opere da cartongessista"`
  - `robimb extract schemas --list` / `robimb extract schemas --show "Opere di pavimentazione" --print-schema`
  - `robimb extract pack --registry pack/v1/registry.json --extractors pack/v1/extractors.json --out-dir data/properties`
  - `robimb extract evaluate --pred outputs/pred.jsonl --gold data/properties/validation/pilota.jsonl --report reports/pilota.md --metrics em,f1,unit-mae`
- **Parametri chiave**
  - `--pack`: directory bundle (default `pack/current`).
  - `--schema`: override del registry (default `data/properties/registry.json`).
  - `--llm-endpoint`, `--llm-model`, `--llm-timeout`, `--llm-max-retries`.
  - `--log-file`: log JSONL (per record `text_id`, step, durata, warnings).
  - `--fail-fast`: interrompe la pipeline al primo errore bloccante.
- **Codici d'errore**
  - `0`: successo.
  - `1`: errore input/validazione schema.
  - `2`: errore LLM (timeout, JSON invalido).
  - `3`: errore scrittura output.
  - `4`: errori parziali (output con warnings / fallback).
- **Esempi uso**
  ```bash
  robimb extract properties --input data/raw/pilota.jsonl --output outputs/pilota_props.jsonl --pack pack/v1 --llm-endpoint https://llm.internal --llm-model gpt-mini --confidence-threshold 0.7
  robimb extract schemas --show "Opere da serramentista" --print-schema
  robimb extract evaluate --pred outputs/pilota_props.jsonl --gold data/properties/validation/pilota.jsonl --report reports/pilota.md --metrics em,f1,unit-mae
  ```

## Rischi & Mitigazioni
| Rischio | Probabilità | Impatto | Mitigazione | Piano di test |
|---------|-------------|---------|-------------|---------------|
| Parser dimensioni non copre formati legacy (es. `1200×600×35`) | Medio | Alto | Regex multiple + normalizzatori unità + fallback QA | `tests/test_parsers_dimensions.py` con ≥80 esempi reali/sintetici |
| QA LLM restituisce JSON invalido | Alto | Medio | Validazione schema, retry, temperature 0, fallback deterministico | Test mocking LLM con risposte errate, assert fallback |
| Matcher brand produce falsi positivi in descrizioni multi-prodotto | Medio | Medio | Dizionari curati + soglie + controllo contesto categoria | Benchmark 200 esempi con precision ≥0.9 |
| Fusione sovrascrive valori affidabili | Basso | Alto | Policy priorità + punteggi confidence + audit log | `tests/test_fuse_validate.py` con scenari conflitto |
| Metriche unit-aware lente su dataset grande | Medio | Medio | Vectorization numpy/pandas + caching parsing | Benchmark su 10k record, target <5s |
| Performance LLM rallenta throughput | Medio | Alto | Batch async, caching, fallback parser-only <0.5s | `tests/perf/test_throughput.py` con scenario 1000 record |

## KPI & Metriche
- Coverage proprietà obbligatorie ≥0.85 sulle sette categorie.
- Exact Match macro-average ≥0.75 (Fase C) → ≥0.85 (Fase D).
- F1 medio su proprietà multi-valore ≥0.80.
- Unit-aware MAE: dimensioni <5 mm; trasmittanza <0.1 W/m²K; Rw <1.5 dB.
- Percentuale null corretti ≥0.9.
- Tempo per 1000 record ≤45s (pipeline completa).
- Tasso errori validazione <2% dopo Fase C.
- Brier score sulle confidence ≤0.15.

## Appendice
```
src/robimb/extraction/
  router.py
  schema_registry.py
  prompts.py
  qa_llm.py
  parsers/
    __init__.py
    numbers.py
    dimensions.py
    units.py
    colors.py
    standards.py
  matchers/
    __init__.py
    brands.py
    materials.py
  normalize.py
  validators.py
  fuse.py
  orchestrator.py

data/properties/
  registry.json
  schema/
    apparecchi_sanitari_accessori.json
    controsoffitti.json
    opere_da_cartongessista.json
    opere_da_falegname.json
    opere_da_serramentista.json
    opere_di_pavimentazione.json
    opere_di_rivestimento.json
  prompts.json
  lexicon/
    brands.txt
    materials.txt
    colors_ral.json
    standards_prefixes.json
  validation/
    pilota.jsonl
  calibration.json

pack/
  v1_limited/
  v1/
    manifest.json
    registry.json
    extractors.json
    prompts.json
    validators.json

docs/
  ROADMAP.md
  ADR/
    0001-schema-first.md
    0002-fusion-strategy.md
  cli_extract.md
  property_extraction.md
```
