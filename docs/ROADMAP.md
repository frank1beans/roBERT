# Sintesi esecutiva
- Audit del pack `pack/v1_limited` completato: gli slot disponibili coprono quasi esclusivamente `marchio`, poche dimensioni (`spessore_mm`, `dimensione_larghezza`, `dimensione_altezza`) e classi prestazionali generiche, confermando la natura regex-first del bundle attuale.
- Definito un registro schema-first (`data/properties/registry.json`) con tre categorie pilota e metadata completi (tipi, unità, enum, required, fonti), collegato a JSON Schema validabili in `data/properties/schema/`.
- Implementati parser deterministici per numeri, unità e dimensioni (`src/robimb/extraction/parsers/`) con 60+ casi di test e metriche EM=1.0 sui set pilota.
- Nuovo skeleton CLI `robimb extract` basato su Typer con sottocomandi `properties`, `schemas`, `pack`, pronto a collegare l’orchestrator e a ispezionare il registry.

## Assunzioni
- Dataset pilota disponibile con almeno 100 descrizioni per ciascuna delle tre categorie selezionate.
- Accesso a un LLM compatto (GPT-4o-mini o equivalente) via API interna con SLA <1s.
- Glossari brand/materiali/standard forniti dagli esperti dominio entro la fine della Fase A.
- Possibilità di eseguire Python ≥3.11, pytest, jsonschema e Typer nell’ambiente CI/DEV.

## Roadmap
### Fase A — Foundations (settimane 1-3)
- **Obiettivi**
  - Inventario completo del pack legacy e definizione schema-first per le categorie pilota.
  - Parser deterministici numeri/unità/dimensioni e CLI skeleton.
- **Attività principali**
  - Audit pack (`pack/v1_limited/*`), redazione ADR schema-first.
  - Creazione `data/properties/registry.json` e JSON Schema pilota.
  - Implementazione parser (`numbers.py`, `units.py`, `dimensions.py`) + test ≥50 casi.
  - Nuovo comando `robimb extract` con sottocomandi `properties` (placeholder orchestrator) e `schemas`.
- **Dipendenze**: accesso pack legacy, conferma categorie pilota, disponibilità esperto dominio per review.
- **Rischi/Mitigazioni**
  - Parser dimensioni non coprono formati reali → raccolta esempi reali e test generativi.
  - Schemi incompleti → review con dominio + ADR.
- **Deliverable/DoD**
  - `data/properties/registry.json` validato con jsonschema, tre JSON Schema in `data/properties/schema/`.
  - Parser con pytest verde e coverage logica ≥90% sui casi mappati.
  - `robimb extract properties --dry-run` funzionante con output JSON di configurazione.

### Fase B — Pipeline ibrida (settimane 4-7)
- **Obiettivi**
  - QA estrattivo JSON-constrained, lexicon matchers, fusione candidati e validazione schema.
- **Attività principali**
  - Implementazione `prompts.py`, `qa_llm.py`, normalizzatori e matchers brand/materiali/colori/standard.
  - `fuse.py`, `validators.py` con Pydantic v2, orchestrator end-to-end e logging strutturato.
  - Dataset golden set (≥30 record/categoria) per regression test.
- **Dipendenze**: completamento Fase A, API LLM funzionante, lexicon iniziali.
- **Rischi/Mitigazioni**
  - LLM produce JSON invalido → validazione schema + retry con temperatura 0.
  - Lessici incompleti → pipeline aggiornamento automatica da CSV e fallback QA.
- **Deliverable/DoD**
  - QA extractor con EM ≥0.75 su golden set.
  - Fusione con log JSONL (source/span/confidence) e CLI `properties` che genera output JSONL.

### Fase C — Qualità & Metriche (settimane 8-10)
- **Obiettivi**
  - Dataset validazione (≥300 record), metriche unit-aware e report automatici.
- **Attività principali**
  - `data/properties/validation/pilota.jsonl`, `src/robimb/utils/metrics.py`, reporting Markdown/HTML.
  - CLI `robimb extract evaluate` con soglie di confidenza e generazione report.
- **Dipendenze**: pipeline stabile (Fase B), dataset etichettato.
- **Rischi/Mitigazioni**
  - Dataset sbilanciato → synthetic data + stratified sampling.
  - Metriche lente → vectorization (numpy/pandas).
- **Deliverable/DoD**
  - Report qualità con coverage ≥0.8 e EM medio ≥0.7.
  - Integrazione CI che esegue eval e pubblica badge.

### Fase D — Hardening (settimane 11-14)
- **Obiettivi**
  - Edge cases (range, ≥, simboli), normalizzazione avanzata, performance e packaging pack `v1`.
- **Attività principali**
  - Estensione parser range/comparatori, normalizzatori RAL e standard UNI/EN, benchmark throughput.
  - Packaging PyPI-ready e documentazione finale (CLI, API, guide).
- **Dipendenze**: metriche consolidate (Fase C), dataset completo.
- **Rischi/Mitigazioni**
  - Degrado performance >45s/1k record → parallel Typer + caching LLM/matchers.
  - Divergenza pack vs codice → CI con checksum e validazione pack.
- **Deliverable/DoD**
  - `pack/v1/` completo con manifest, registry, prompts e lexicon.
  - Benchmark `tests/perf/test_throughput.py` ≤45s/1000 record.
  - Documentazione aggiornata (`docs/cli_extract.md`, `docs/property_extraction.md`).

## Backlog
- **A1. Audit pack v1_limited** — File: `pack/v1_limited/*`, `docs/ROADMAP.md`. DoD: tabella asset + gap in Appendice; review TL. Stima: 6h.
- **A2. ADR schema-first** — File: `docs/ADR/0001-schema-first.md`. DoD: ADR approvato (pro/contro, decisione). Stima: 5h.
- **A3. Registry pilota** — File: `data/properties/registry.json`. DoD: JSON validato, categorie pilota presenti, metadati completi. Stima: 8h.
- **A4. JSON Schema categorie** — File: `data/properties/schema/*.json`. DoD: validazione via `jsonschema` + test di caricamento. Stima: 10h.
- **A5. Schema registry loader** — File: `src/robimb/extraction/schema_registry.py`. DoD: API `load_registry` con cache + test `tests/test_schema_registry.py`. Stima: 8h.
- **A6. Validators Pydantic** — File: `src/robimb/extraction/validators.py`. DoD: `validate_properties` ritorna errori strutturati, coverage ≥90%. Stima: 9h.
- **A7. Parser numerici** — File: `src/robimb/extraction/parsers/numbers.py`. DoD: funzioni `parse_number_it`/`extract_numbers`, pytest ≥30 casi. Stima: 8h.
- **A8. Parser dimensioni** — File: `src/robimb/extraction/parsers/dimensions.py`. DoD: supporto formati `90x210`, `0,90×2,10`, `L90 H210`; ≥25 casi, EM ≥0.9. Stima: 12h.
- **A9. Parser unità** — File: `src/robimb/extraction/parsers/units.py`. DoD: normalizzazione mm/cm/m/m²/m³/kN/m², pytest ≥15 casi. Stima: 7h.
- **A10. CLI skeleton** — File: `src/robimb/cli/extract.py`, `src/robimb/cli/main.py`. DoD: `robimb extract properties --help` e `--dry-run` attivi. Stima: 6h.
- **A11. Logging strutturato** — File: `src/robimb/utils/logging.py`, orchestrator. DoD: log JSONL con trace-id testato. Stima: 5h.
- **B1. Prompt library** — File: `src/robimb/extraction/prompts.py`, `data/properties/prompts.json`. DoD: template LLM con test rendering. Stima: 7h.
- **B2. QA LLM adapter** — File: `src/robimb/extraction/qa_llm.py`. DoD: retry/backoff + test mock. Stima: 10h.
- **B3. Lexicon marchi/materiali** — File: `data/properties/lexicon/*.txt`. DoD: ≥200 voci, check duplicati. Stima: 6h.
- **B4. Matcher marchi** — File: `src/robimb/extraction/matchers/brands.py`. DoD: precision ≥0.9 su 200 esempi. Stima: 8h.
- **B5. Matcher materiali** — File: `src/robimb/extraction/matchers/materials.py`. DoD: sinonimi/lemmatizzazione, EM ≥0.8. Stima: 8h.
- **B6. Parser colori RAL** — File: `src/robimb/extraction/parsers/colors.py`, `data/properties/lexicon/colors_ral.json`. DoD: mapping completo, test 30 codici. Stima: 7h.
- **B7. Parser standard** — File: `src/robimb/extraction/parsers/standards.py`. DoD: recall ≥0.95 su 50 esempi UNI/EN. Stima: 9h.
- **B8. Fusione candidati** — File: `src/robimb/extraction/fuse.py`. DoD: policy configurabile, test conflitti. Stima: 12h.
- **B9. Orchestrator** — File: `src/robimb/extraction/orchestrator.py`. DoD: pipeline completa con config Typer, integration test 30 record. Stima: 10h.
- **B10. Trace logging** — File: `src/robimb/extraction/fuse.py`, `orchestrator.py`. DoD: output include `source/span/confidence`, snapshot test. Stima: 6h.
- **B11. Pack prompts/manifest** — File: `pack/v1/*`. DoD: manifest valido via jsonschema. Stima: 5h.
- **C1. Dataset pilota** — File: `data/properties/validation/pilota.jsonl`. DoD: ≥300 record bilanciati ±10%. Stima: 12h.
- **C2. Metriche unit-aware** — File: `src/robimb/utils/metrics.py`. DoD: `exact_match`, `unit_aware_mae`, test numerici. Stima: 9h.
- **C3. Report qualità** — File: `src/robimb/reporting/properties_report.py`, `docs/property_extraction_report_template.md`. DoD: report Markdown/HTML rigenerabile. Stima: 10h.
- **C4. CLI evaluate** — File: `src/robimb/cli/extract.py`. DoD: sottocomando `evaluate` (estensione) con test CLI. Stima: 7h.
- **C5. Confidence calibration** — File: `src/robimb/extraction/fuse.py`, `data/properties/calibration.json`. DoD: curva precision-recall + soglie salvate. Stima: 8h.
- **C6. Coverage analytics** — File: `src/robimb/reporting/coverage.py`. DoD: grafico coverage property/category. Stima: 6h.
- **C7. CI metrics** — File: `.github/workflows/qa_properties.yml`. DoD: workflow con badge. Stima: 8h.
- **D1. Parser range/comparatori** — File: `src/robimb/extraction/parsers/dimensions.py`. DoD: supporto `≥`, `da...a`, `±`, test 60 esempi. Stima: 10h.
- **D2. Normalizzatori avanzati** — File: `src/robimb/extraction/normalize.py`. DoD: funzioni `normalize_dimension_range`, `normalize_standard`, test 40 casi. Stima: 8h.
- **D3. Validator messaggi azionabili** — File: `src/robimb/extraction/validators.py`. DoD: errori con codici/suggerimenti, snapshot test. Stima: 6h.
- **D4. Performance benchmark** — File: `tests/perf/test_throughput.py`. DoD: ≤45s/1000 record. Stima: 9h.
- **D5. Packaging** — File: `pyproject.toml`, `src/robimb/__init__.py`. DoD: wheel installabile, CLI disponibile. Stima: 6h.
- **D6. Documentazione finale** — File: `docs/cli_extract.md`, `docs/property_extraction.md`, `docs/ROADMAP.md`. DoD: guide aggiornate, review tech writer. Stima: 7h.
- **D7. Pack v1** — File: `pack/v1/*`. DoD: checksum + manifest aggiornati. Stima: 8h.
- **D8. Knowledge transfer** — File: `docs/KT/property_extraction_slides.pdf`. DoD: sessione registrata, feedback ≥4/5. Stima: 5h.

## Matrice Proprietà×Categoria (pilot)
### Porte HPL
| Property ID | Tipo | Unità | Regole di validazione | Esempio testo → output |
|-------------|------|-------|------------------------|------------------------|
| `larghezza_anta_mm` | float | mm | 600 ≤ val ≤ 1600 | “anta 90×210 cm” → 900 |
| `altezza_anta_mm` | float | mm | 1800 ≤ val ≤ 3000 | “porta 0,90×2,10 m” → 2100 |
| `spessore_anta_mm` | float | mm | 30 ≤ val ≤ 80 | “spessore 45 mm” → 45 |
| `classe_rei` | enum | — | {REI30, REI60, REI90, REI120} | “REI 60” → REI60 |
| `finitura_superficie` | string | — | stringa 3-40 char, supporto RAL | “RAL 9010” → “RAL 9010” |
| `materiale_telaio` | enum | — | {alluminio, acciaio, legno, pvc} | “telaio in acciaio” → “acciaio” |
| `dotazione_accessori` | list[str] | — | 1-5 voci, 2-80 char | “accessori: chiudiporta, spioncino” |
| `marchio` | string | — | 2-80 char | “maniglia Cisa” → “Cisa” |

### Controsoffitti metallici
| Property ID | Tipo | Unità | Regole di validazione | Esempio |
|-------------|------|-------|-----------------------|---------|
| `tipologia_modulo` | enum | — | {doghe, pannelli, grigliato} | “controsoffitto a pannelli” |
| `dimensione_modulo_mm` | list[float] | mm | 2-3 valori, 50–2000 mm | “600×600 mm” → [600,600] |
| `materiale_pannello` | enum | — | {acciaio, alluminio, acciaio_microforato} | “alluminio microforato” |
| `trattamento_superficiale` | enum | — | {verniciato, anodizzato, polveri_epossidiche} | “verniciato polveri epossidiche” |
| `prestazione_acustica_db` | float | dB | 20 ≤ val ≤ 50 | “isolamento 40 dB” → 40 |
| `classe_reazione_fuoco` | enum | — | {A1, A2, B-s1,d0} | “classe A1” |
| `spessore_pannello_mm` | float | mm | 0.4 ≤ val ≤ 1.2 | “spessore 0,7 mm” |
| `standard_riferimento` | list[str] | — | 0-6 stringhe 4-40 char | “UNI EN 13964” |

### Pavimenti sopraelevati
| Property ID | Tipo | Unità | Regole di validazione | Esempio |
|-------------|------|-------|-----------------------|---------|
| `spessore_pannello_mm` | float | mm | 25 ≤ val ≤ 60 | “pannello 30 mm” |
| `altezza_struttura_regolabile_mm` | range | mm | array 2 valori, 80 ≤ min ≤ 1200 | “struttura 120-600 mm” → [120,600] |
| `carico_utile_kn_m2` | float | kN/m² | 2 ≤ val ≤ 10 | “carico 4 kN/m²” |
| `rivestimento_superiore` | enum | — | {gres, laminato, pvc, moquette} | “finitura gres” |
| `materiale_nucleo` | enum | — | {solfato_calcio, truciolare, alluminio} | “nucleo solfato di calcio” |
| `classe_reazione_fuoco` | enum | — | {Bfl-s1, Cfl-s1} | “classe Bfl-s1” |
| `resistenza_umidita_percent` | float | % | 0 ≤ val ≤ 100 | “resistenza 85%” |
| `marchio` | string | — | 2-80 char | “sistema Nesite” |

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
  {"text_id":"porte_001","categoria":"Porte HPL","text":"Porta REI 60 90x210 cm, finitura RAL 9010, telaio acciaio, maniglia Cisa antipanico."}
  {"text_id":"contro_005","categoria":"Controsoffitti metallici","text":"Pannelli metallici 600x600 mm in alluminio microforato spessore 0,7 mm, verniciati polveri epossidiche, classe A1, UNI EN 13964."}
  {"text_id":"pav_010","categoria":"Pavimenti sopraelevati","text":"Sistema pavimento Nesite, pannello 30 mm solfato di calcio, finitura gres, struttura regolabile 120-600 mm, carico 4 kN/m², Bfl-s1."}
  ```

## CLI Spec (`robimb extract`)
- **Comando principale**: `robimb extract` (Typer multi-app).
- **Sottocomandi**
  - `properties`: esegue (o simula) l’orchestrazione dell’estrazione.
    - Opzioni principali: `--input`, `--output`, `--pack`, `--schema`, `--llm-endpoint`, `--llm-model`, `--llm-timeout`, `--llm-max-retries`, `--category-filter`, `--confidence-threshold`, `--batch-size`, `--max-workers`, `--log-file`, `--fail-fast`, `--dry-run`.
    - Output (oggi): JSON di configurazione con `status: pipeline_not_implemented`; DoD futuro: scrittura JSONL conforme al contratto.
  - `schemas`: ispeziona il registry; flag `--list`, `--show`, `--print-schema`.
  - `pack`: compatibilità legacy per spacchettare registry/extractors.
- **Codici d’errore attesi**
  - `0`: esecuzione corretta/dry-run.
  - `1`: validazione input fallita o orchestrator non disponibile.
  - `2`: errore LLM (timeout/JSON invalido) — da implementare con orchestrator.
  - `3`: errore scrittura output.
  - `4`: run conclusa con warnings (output parziale).
- **Esempi**
  ```bash
  robimb extract properties --input data/raw/pilota.jsonl --output outputs/pilota.jsonl --pack pack/v1 --schema data/properties/registry.json --llm-endpoint https://llm.internal --llm-model gpt-mini --confidence-threshold 0.7 --dry-run
  robimb extract schemas --show "Porte HPL" --print-schema
  robimb extract pack --registry pack/v1/registry.json --extractors pack/v1/extractors.json --out-dir data/properties
  ```

## Rischi & Mitigazioni
| Rischio | Prob. | Impatto | Mitigazione | Piano di test |
|---------|-------|---------|-------------|---------------|
| Parser dimensioni fallisce su formati misti (L/H, simboli ×) | Medio | Alto | Heuristics + dataset ampliato | `tests/test_parsers_dimensions.py` con ≥25 casi reali |
| LLM restituisce JSON invalido | Alto | Medio | Validazione schema + retry, fallback deterministico | Test mock con output errati + check fallback |
| Matcher marchi produce falsi positivi | Medio | Medio | Dizionari curati + soglie + cross-check schema | Benchmark 200 esempi, precision ≥0.9 |
| Validazione schema troppo rigida → drop record | Medio | Medio | Messaggi azionabili, modalità `allow_partial` | Test su payload parziali, snapshot errori |
| Throughput insufficiente su 1k record | Medio | Alto | Async batching LLM, caching, parallel workers | Benchmark `tests/perf/test_throughput.py` |
| Divergenza pack/code | Basso | Alto | CI con checksum pack e validazione `jsonschema` | Workflow `qa_properties.yml` |

## KPI & Metriche
- Coverage proprietà obbligatorie ≥0.85 (macro-average per categoria).
- Exact Match per proprietà ≥0.75 (target Fase B), ≥0.85 (Fase D).
- F1 proprietà multivalore ≥0.80.
- Unit-aware MAE: errore medio dimensioni <5 mm, carichi <0.2 kN/m².
- % null corretti ≥0.9.
- Tempo elaborazione ≤45s/1000 record (hardware target Fase D).
- Tasso errori validazione <2% record dopo Fase C.
- Brier score confidenza ≤0.15.

## Appendice
### Audit pack `v1_limited`
| Macro-categoria | Slot globali | Slot specifici | Note |
|-----------------|--------------|----------------|------|
| Opere da cartongessista | `marchio`, `tipologia_lastra`, `spessore_mm`, `classe_ei`, `classe_reazione_al_fuoco`, `presenza_isolante`, `stratigrafia_lastre` | `tipologia_accessorio` (solo per Accessori) | Focus su brand/materiale; nessuna unità convertita, niente tracing span |
| Opere di rivestimento | `marchio`, `materiale`, `spessore_mm`, `classe_reazione_al_fuoco`, `resa_mq`, `coefficiente_termico` | `tipologia_rivestimento`, `classe_antiscivolo`, `capacita_serbatoio_l` | Proprietà non normalizzate, mismatch con esigenze categorie pilota |
| Opere da falegname | `marchio`, `essenza`, `dimensione_larghezza`, `dimensione_altezza`, `tipologia_apertura` | `numero_ante`, `tipologia_lamelle`, `tipologia_porta` | Dimensioni senza unità esplicite, nessun span tracking |

Altri componenti pack:
- `extractors.json`: ~320 pattern regex focalizzati su brand, materiali, pochi valori dimensionali; nessun parsing unità.
- `validators.json`, `formulas.json`, `views.json`, `templates.json`, `profiles.json`, `contexts.json`: strutture vuote o placeholder.
- `manifest.json`: schema `manifest/v1` con sola referenza a registry/extractors.

### Folder tree (target)
```
src/robimb/extraction/
  schema_registry.py
  parsers/
    __init__.py
    numbers.py
    units.py
    dimensions.py
  (prossimi step: qa_llm.py, matchers/, fuse.py, orchestrator.py)

data/properties/
  registry.json
  schema/
    porte_hpl.json
    controsoffitti_metallici.json
    pavimenti_sopraelevati.json
  (next: prompts.json, lexicon/, validation/, calibration.json)

docs/
  ROADMAP.md
  ADR/
    0001-schema-first.md (da produrre in Fase A)
```

### Template schema categoria
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "CategoriaPlaceholder",
  "type": "object",
  "required": ["category", "properties"],
  "properties": {
    "category": {"const": "CategoriaPlaceholder"},
    "properties": {
      "type": "object",
      "required": [],
      "properties": {},
      "additionalProperties": false
    }
  },
  "additionalProperties": false
}
```

### Esempio output atteso (Porte HPL)
```json
{
  "text_id": "porte_001",
  "categoria": "Porte HPL",
  "properties": {
    "larghezza_anta_mm": {"value": 900, "unit": "mm", "source": "parser", "span": [15, 21], "confidence": 0.92},
    "altezza_anta_mm": {"value": 2100, "unit": "mm", "source": "parser", "span": [22, 28], "confidence": 0.92},
    "classe_rei": {"value": "REI60", "source": "parser", "span": [6, 12], "confidence": 0.95},
    "materiale_telaio": {"value": "acciaio", "source": "matcher", "span": [44, 50], "confidence": 0.90},
    "finitura_superficie": {"value": "RAL 9010", "source": "parser", "span": [32, 39], "confidence": 0.88},
    "dotazione_accessori": {"value": ["maniglia antipanico"], "source": "qa_llm", "span": [52, 72], "confidence": 0.80},
    "marchio": {"value": "Cisa", "source": "matcher", "span": [73, 77], "confidence": 0.85}
  },
  "warnings": [],
  "errors": []
}
```
