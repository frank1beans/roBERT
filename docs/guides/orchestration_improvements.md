# Miglioramenti Orchestrazione: Rules → Matchers → QA → LLM

## Problemi Attuali

Analizzando i risultati di estrazione reali, emergono questi pattern:

### Pattern di Errore

1. **60% documenti**: Proprietà obbligatorie mancanti
2. **40% casi**: LLM estrae parzialmente (alcune prop sì, altre no)
3. **Inferenze errate**: Materiali dedotti male (es. ceramica per metallo)
4. **Mancanza contesto**: Non usa conoscenza dominio (es. "doghe" → legno)

### Esempio Problematico

```json
{
  "text": "Miscelatore monocomando per lavabo tipo Grohe...
           cartuccia a dischi ceramici...",
  "properties": {
    "materiale": {"value": "ceramica", "source": "qa_llm"}  // SBAGLIATO!
  }
}
```

**Problema**: LLM vede "ceramica" nel testo (cartuccia ceramica) e lo usa come materiale del miscelatore (che è metallo).

---

## 🎯 Soluzioni Proposte

### 1. Orchestrazione Multi-Pass con Context Awareness

#### Architettura Attuale
```
Input → Rules → Matchers → QA → LLM → Fuse → Output
        (1 pass)
```

#### Architettura Proposta
```
Input → Pass 1: Extraction
        ├─→ Rules (deterministico)
        ├─→ Matchers (lessici)
        └─→ Gap Analysis

      → Pass 2: Intelligent Filling
        ├─→ QA Encoder (se disponibile)
        └─→ LLM (solo per gap, con context)

      → Pass 3: Validation & Inference
        ├─→ Domain Rules (es. "doghe" → legno)
        ├─→ Consistency Check
        └─→ Confidence Recalibration

      → Output
```

---

### 2. Context-Aware LLM Prompts

#### Problema Attuale
Prompt generico per tutte le proprietà:

```
Testo: {text}
Domanda: Estrai {property}
Schema: {schema}
```

#### Soluzione: Prompt Specifici per Tipo

**Per Materiali:**
```
Testo: {text}
Domanda: Qual è il materiale PRINCIPALE dell'oggetto descritto?

Istruzioni:
- NON confondere il materiale dell'oggetto con materiali di componenti
- "cartuccia ceramica" → il miscelatore è in METALLO
- "doghe" → probabilmente LEGNO
- "cristallo/vetro temperato" → VETRO
- Se menziona "cromato/acciaio/inox" → METALLO

Rispondi in JSON: {{"value": "...", "confidence": 0.0-1.0}}
```

**Per Dimensioni:**
```
Testo: {text}
Domanda: Estrai dimensioni (lunghezza x larghezza x altezza) in mm.

Formato atteso: "60x60" significa 60mm x 60mm
Se manca altezza, lascia null (NON copiare lunghezza/larghezza)

Rispondi: {{"lunghezza": X, "larghezza": Y, "altezza": Z_or_null}}
```

---

### 3. Domain Knowledge Injection

#### Rules Euristiche per Inferenza

```python
DOMAIN_RULES = {
    "materiale": {
        "keywords": {
            "doghe": "legno",
            "cristallo": "vetro",
            "temperato": "vetro",
            "inox|acciaio": "metallo_acciaio",
            "cromato": "metallo_cromato",
            "pvc": "plastica_pvc",
            "gres": "ceramica_gres",
            "cartongesso": "gesso"
        },
        "category_defaults": {
            "miscelatore": "metallo",
            "rubinetto": "metallo",
            "box_doccia": "vetro",
            "specchio": "vetro",
            "maniglione": "metallo"
        }
    },
    "tipologia_installazione": {
        "keywords": {
            "a parete|parete": "a_parete",
            "sospeso": "sospesa",
            "pavimento": "a_pavimento",
            "incasso": "ad_incasso"
        }
    }
}
```

**Applicazione:**
1. Prima prova Rules/Matchers
2. Se falliscono, prova keywords
3. Se ancora null, prova category default
4. Solo alla fine, chiedi a LLM

---

### 4. Gap-Aware LLM Calling

#### Strategia Attuale
LLM chiamato per TUTTE le proprietà in parallelo (anche quelle già estratte).

#### Strategia Proposta

```python
def orchestrate_extraction(text, category, schema):
    # Pass 1: Deterministic
    results = {}
    results.update(apply_rules(text))
    results.update(apply_matchers(text))
    results.update(apply_domain_heuristics(text, category))

    # Gap Analysis
    required_props = schema.get_required_properties()
    missing = [p for p in required_props if not results.get(p)]

    # Pass 2: Fill Gaps (smart)
    if missing:
        # 2a: QA Encoder (veloce, per proprietà strutturate)
        if qa_encoder_available:
            qa_results = qa_encoder.extract(text, missing)
            results.update(qa_results)
            missing = [p for p in missing if not results.get(p)]

        # 2b: LLM (costoso, solo per remaining gaps)
        if missing and llm_available:
            # Batch LLM call con context
            llm_prompt = build_gap_filling_prompt(
                text=text,
                category=category,
                missing_properties=missing,
                already_extracted=results
            )
            llm_results = llm.extract(llm_prompt)
            results.update(llm_results)

    # Pass 3: Validation & Consistency
    results = validate_and_fix(results, schema, text)

    return results
```

**Vantaggi:**
- ✅ Riduce chiamate LLM (solo per gap)
- ✅ LLM ha context di cosa è già estratto
- ✅ Più veloce (rules/matchers first)
- ✅ Più economico (meno token)

---

### 5. Cross-Property Validation

#### Problema
LLM estrae proprietà indipendentemente, senza controllare coerenza.

#### Soluzione: Validation Rules

```python
VALIDATION_RULES = {
    "apparecchi_sanitari_accessori": {
        "consistency": [
            {
                "if": {"text_contains": "miscelatore|rubinetto"},
                "then": {"materiale": {"should_be": "metallo*"}},
                "fix": "override_with_domain_default"
            },
            {
                "if": {"materiale": "vetro"},
                "then": {"properties": ["dimensione_*"]},
                "message": "Vetro richiede dimensioni specificate"
            },
            {
                "if": {"text_contains": "a parete"},
                "then": {"tipologia_installazione": {"should_be": "a_parete"}},
                "fix": "auto_correct"
            }
        ],
        "dimensions": {
            "check": "if lunghezza == altezza AND text NOT contains 'quadrato|cubo'",
            "action": "flag_suspicious",
            "confidence_penalty": 0.2
        }
    }
}
```

---

### 6. Confidence Recalibration

#### Problema
LLM restituisce sempre confidence = 1.0, anche quando sbaglia.

#### Soluzione: Calibrazione Post-Estrazione

```python
def recalibrate_confidence(property_value, context):
    base_confidence = property_value.get("confidence", 1.0)

    # Penalità per inferenze indirette
    if property_value["source"] == "qa_llm":
        # Controlla se il valore appare letteralmente nel testo
        if property_value["value"].lower() not in context["text"].lower():
            base_confidence *= 0.8  # Inferito, non esplicito

    # Boost per concordanza multi-sorgente
    if multiple_sources_agree(property_value):
        base_confidence = min(1.0, base_confidence * 1.2)

    # Penalità per valori sospetti
    if is_suspicious(property_value, context):
        base_confidence *= 0.5

    return base_confidence
```

---

### 7. Prompt Engineering per Proprietà Mancanti

#### Template Ottimizzato

```python
PROPERTY_SPECIFIC_PROMPTS = {
    "materiale": """
Testo: {text}

Identifica il MATERIALE PRINCIPALE dell'oggetto.

Regole:
1. Cerca materiali dell'OGGETTO, non di sue parti (es. "cartuccia ceramica" nel miscelatore → il miscelatore è METALLO, non ceramica)
2. Usa conoscenza dominio:
   - Miscelatori/Rubinetti → quasi sempre METALLO (acciaio/ottone/cromato)
   - Box doccia "cristallo" → VETRO
   - Seggiolino "doghe" → LEGNO
   - Griglia "AISI 304" → ACCIAIO_INOX

Valori validi: {valid_values}

Se NON trovi esplicito nel testo, usa conoscenza dominio.
Rispondi: {{"value": "...", "confidence": 0.0-1.0, "reasoning": "breve spiegazione"}}
""",

    "dimensioni": """
Testo: {text}

Estrai dimensioni in millimetri.

Formati comuni:
- "60x60" → lunghezza=600mm, larghezza=600mm
- "90x70 h190" → lunghezza=900mm, larghezza=700mm, altezza=1900mm
- "235x70 cm" → lunghezza=2350mm, larghezza=700mm

IMPORTANTE:
- Se altezza non specificata → lascia null (NON copiare lunghezza)
- Converti sempre in mm (cm → mm: moltiplica x10)

Rispondi: {{
  "lunghezza": number_or_null,
  "larghezza": number_or_null,
  "altezza": number_or_null,
  "confidence": 0.0-1.0
}}
"""
}
```

---

## 📈 Implementazione Prioritaria

### Quick Wins (1-2 giorni)

1. **Domain Heuristics** - Regole semplici per materiali comuni
   ```python
   # In extraction/orchestrator.py
   def apply_domain_heuristics(text, category):
       # Implementa DOMAIN_RULES sopra
   ```

2. **Prompt Engineering** - Template specifici per proprietà critiche
   ```python
   # In extraction/prompts.json
   # Aggiungi prompt ottimizzati
   ```

3. **Confidence Recalibration** - Penalità per inferenze indirette
   ```python
   # In extraction/fuse.py
   def recalibrate_confidence(prop, text):
       # Check se valore appare letteralmente
   ```

### Medium Term (1 settimana)

4. **Gap-Aware Orchestration** - LLM solo per proprietà mancanti
   ```python
   # Refactor orchestrator.py per multi-pass
   ```

5. **Cross-Property Validation** - Consistency checks
   ```python
   # In extraction/validators.py
   # Aggiungi VALIDATION_RULES
   ```

### Long Term (1-2 settimane)

6. **Adaptive Strategy Selection** - Scegli strategia per categoria
   ```python
   # Router che decide rules vs QA vs LLM based on category
   ```

7. **Feedback Loop** - Impara da errori
   ```python
   # Log errori comuni → aggiorna heuristics/prompts
   ```

---

## 🧪 Testing Strategy

### Baseline Test
```bash
# Current performance
robimb extract properties \
  --input test_set.jsonl \
  --output baseline.jsonl \
  --sample 100

python scripts/analysis/extraction_results.py baseline.jsonl > baseline_metrics.txt
```

### A/B Testing
```bash
# Test each improvement
# 1. Heuristics only
# 2. Heuristics + better prompts
# 3. Heuristics + prompts + gap-aware
# 4. Full pipeline

# Compare metrics
```

### Success Metrics
- **Coverage**: % proprietà obbligatorie estratte (target: >90%)
- **Accuracy**: % valori corretti (manual check su sample)
- **Cost**: Token LLM usati (target: -30%)
- **Speed**: Tempo per documento (target: invariato)

---

## 🎯 ROI Stimato

| Improvement | Effort | Impact | Cost Reduction |
|-------------|--------|--------|----------------|
| Domain Heuristics | 🟢 Low | 🟢 High | -20% LLM calls |
| Better Prompts | 🟢 Low | 🟡 Medium | +10% accuracy |
| Gap-Aware LLM | 🟡 Medium | 🟢 High | -40% LLM calls |
| Validation Rules | 🟡 Medium | 🟡 Medium | +15% accuracy |
| Confidence Recal | 🟢 Low | 🟡 Medium | Better filtering |

**Totale stimato**: -50% costi LLM, +25% accuracy

---

## 📝 Next Steps

1. **Implementa Domain Heuristics** (oggi)
2. **Testa su 100 documenti** (domani)
3. **Itera su prompts** basato su errori (questa settimana)
4. **Refactor orchestrator** per gap-aware (prossima settimana)

Vuoi che inizi con l'implementazione?
