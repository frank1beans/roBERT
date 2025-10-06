# Changelog

Tutte le modifiche rilevanti al progetto roBERT saranno documentate in questo file.

## [0.2.0] - 2025-10-02

### Riorganizzazione Repository

#### ✨ Aggiunte

- **Documentazione completa CLI**
  - `docs/commands/overview.md`: Panoramica comandi
  - `docs/commands/extract.md`: Documentazione estrazione
  - `docs/commands/convert.md`: Documentazione conversione
  - `docs/commands/train.md`: Documentazione training
  - `docs/commands/evaluate.md`: Documentazione valutazione
  - `docs/commands/pack.md`: Documentazione knowledge pack
  - `docs/commands/config.md`: Documentazione configurazione

- **Documentazione architettura**
  - `ARCHITECTURE.md`: Overview tecnico completo del sistema
  - `scripts/README.md`: Documentazione script di supporto

- **Struttura documentazione**
  - `docs/commands/`: Documentazione comandi CLI
  - `docs/guides/`: Guide tecniche
  - `docs/development/`: Guide sviluppo

#### 🔄 Modifiche

- **Scripts riorganizzati per categoria**
  - `scripts/analysis/`: Script di analisi
    - `analyze_extraction.py` → `extraction_results.py`
    - `analyze_dataset.py` → `dataset_basic.py`
    - `analyze_dataset_detailed.py` → `dataset_detailed.py`
  - `scripts/testing/`: Script di testing
    - Spostati `sanity_check.py` e `test_llm_integration.sh`
  - `scripts/setup/`: Script di setup
    - Spostato `setup_gpt4mini.ps1`

- **README.md completamente riscritto**
  - Aggiunta struttura progetto chiara
  - Workflow tipico passo-passo
  - Documentazione Makefile
  - Esempi pratici per ogni comando
  - Sezione configurazione e performance

- **Makefile aggiornato**
  - Path corretti per script riorganizzati
  - `analyze` → `scripts/analysis/extraction_results.py`
  - `llm-integration-test` → `scripts/testing/test_llm_integration.sh`

- **.gitignore migliorato**
  - Aggiunta coverage e testing artifacts
  - Migliorata gestione outputs e modelli
  - Pulizia duplicati

#### 🗑️ Rimozioni

- `orchestrator_backup.py`: File vuoto rimosso
- `REORGANIZATION_PLAN.md`: File temporaneo rimosso
- `execute_reorganization.sh`: Script obsoleto rimosso

#### 📁 Spostamenti

- `docs/production_resource_setup.md` → `docs/guides/production_resource_setup.md`

### Miglioramenti Manutenibilità

- **Navigazione migliorata**: Struttura chiara con README in ogni directory principale
- **Documentazione completa**: Ogni comando CLI documentato con esempi
- **Onboarding facilitato**: ARCHITECTURE.md per nuovi sviluppatori
- **Scripts organizzati**: Raggruppamento logico per funzionalità

## [0.1.0] - Versione Precedente

Versione iniziale del progetto con:
- Pipeline estrazione ibrida (Rules + QA + LLM)
- Training modelli label e gerarchici
- CLI base con comandi extract, convert, train, evaluate, pack
- Registry risorse e schema
- Parsers e matchers per proprietà BIM
