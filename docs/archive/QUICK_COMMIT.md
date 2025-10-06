# 🚀 Quick Commit Guide

Guida rapida per committare tutte le modifiche.

## ✅ Cosa Verrà Committato

### File Modificati (4)
- `.gitignore` - Migliorato (coverage, models, logs)
- `Makefile` - Path aggiornati per script riorganizzati
- `README.md` - Completamente riscritto (+700% contenuto)
- `src/robimb/cli/extract.py` - Fix bug Typer Literal

### File Eliminati (10)
- `REORGANIZATION_PLAN.md`
- `execute_reorganization.sh`
- `src/robimb/extraction/orchestrator_backup.py`
- `docs/production_resource_setup.md` (spostato)
- Scripts spostati/rinominati (6 file)

### Nuovi File (30+)

**Documentazione (8):**
- `ARCHITECTURE.md`
- `BUGFIX.md`
- `CHANGELOG.md`
- `COMMIT_GUIDE.md`
- `QUICKSTART_GPT4MINI.md`
- `QUICKSTART_GPT4MINI_WINDOWS.md`
- `REFACTORING_SUMMARY.md`
- `RIEPILOGO_FINALE.md`

**CLI Docs (7):**
- `docs/commands/overview.md`
- `docs/commands/extract.md`
- `docs/commands/convert.md`
- `docs/commands/train.md`
- `docs/commands/evaluate.md`
- `docs/commands/pack.md`
- `docs/commands/config.md`

**Guides (1):**
- `docs/guides/production_resource_setup.md` (spostato)

**Examples (3):**
- `examples/README.md`
- `examples/llm_server_example.py`
- `examples/llm_server_gpt4mini.py`

**Scripts (7):**
- `scripts/README.md`
- `scripts/analysis/dataset_basic.py` (rinominato)
- `scripts/analysis/dataset_detailed.py` (rinominato)
- `scripts/analysis/extraction_results.py` (spostato)
- `scripts/setup/README.md`
- `scripts/setup/start_server.ps1`
- `scripts/setup/extract.ps1`

**Config (2):**
- `.env.example`
- `scripts/testing/` e `scripts/setup/` directories

---

## 📝 Comandi Git

### Opzione 1: Commit Tutto (Raccomandato)

```bash
# Stage tutti i cambiamenti
git add -A

# Verifica cosa verrà committato
git status

# Commit con messaggio dettagliato
git commit -m "refactor: riorganizzazione completa repository e integrazione GPT-4o-mini

BREAKING CHANGES: Nessuno (solo riorganizzazione e documentazione)

Modifiche principali:
- Documentazione completa CLI (7 nuovi file in docs/commands/)
- ARCHITECTURE.md con overview tecnico dettagliato
- Server LLM: mock e GPT-4o-mini completamente funzionanti
- Script PowerShell ottimizzati per Windows
- Riorganizzazione scripts per categoria (analysis/, testing/, setup/)
- README riscritto con quick start e workflow completi
- Fix bug critici: Typer Literal, PowerShell encoding, variabili riservate

Nuove funzionalità:
- Server GPT-4o-mini (examples/llm_server_gpt4mini.py)
- Server mock per testing (examples/llm_server_example.py)
- Script PowerShell automation (start_server.ps1, extract.ps1)
- Quick start guides (generale + Windows)
- Documentazione completa ogni comando CLI

File rimossi:
- orchestrator_backup.py (vuoto)
- REORGANIZATION_PLAN.md (temporaneo)
- execute_reorganization.sh (obsoleto)

Dettagli: vedere RIEPILOGO_FINALE.md e CHANGELOG.md

Co-Authored-By: Claude <noreply@anthropic.com>"

# Verifica commit
git log -1 --stat

# Push (quando pronto)
git push origin main
```

### Opzione 2: Commit Incrementale

```bash
# 1. Commit documentazione
git add docs/ *.md
git commit -m "docs: aggiungi documentazione completa CLI e guide"

# 2. Commit server LLM
git add examples/ .env.example
git commit -m "feat: aggiungi server LLM (mock e GPT-4o-mini)"

# 3. Commit script PowerShell
git add scripts/
git commit -m "feat: aggiungi script PowerShell automation"

# 4. Commit riorganizzazione
git add -u
git commit -m "refactor: riorganizza scripts e docs"

# 5. Commit bug fix
git add src/robimb/cli/extract.py
git commit -m "fix: risolvi bug Typer Literal type error"

# 6. Commit config files
git add .gitignore Makefile README.md
git commit -m "chore: aggiorna configurazione e README"

# Push tutti i commit
git push origin main
```

---

## 🔍 Verifica Pre-Commit

### 1. Test Base

```powershell
# Verifica CLI funziona
robimb --help
robimb extract --help

# Test server mock
python examples/llm_server_example.py --port 8000
# CTRL+C per fermare

# Test script PowerShell
.\scripts\setup\start_server.ps1
# CTRL+C per fermare
```

### 2. Verifica File

```bash
# Nessun file sensibile incluso
git status | grep -i ".env"  # Deve mostrare solo .env.example
git status | grep -i "api"   # Nessun file con API keys

# Controlla dimensione commit
git diff --stat
```

### 3. Lint Check (Opzionale)

```bash
# Python
flake8 src/robimb/cli/extract.py
pylint src/robimb/cli/extract.py

# PowerShell
# Apri in VSCode e controlla warnings
```

---

## ⚠️ Note Importanti

### File da NON Committare

✅ Già in `.gitignore`:
- `.env` (API keys)
- `outputs/*.jsonl`
- `*.log`
- `.llm_cache/`
- `__pycache__/`

### Breaking Changes

❌ **Nessuno!** Questa è solo riorganizzazione e documentazione.

Il codice esistente in `src/robimb/` è invariato (tranne 1 bug fix).

### Backward Compatibility

✅ **Preservata**:
- CLI rimane identica
- API interna invariata
- Makefile targets invariati (path aggiornati ma nomi uguali)

---

## 🎯 Dopo il Commit

### 1. Verifica su GitHub

```bash
# Dopo push, vai su:
https://github.com/atipiqal/roBERT

# Controlla:
- README visualizzato correttamente
- Link funzionanti
- Documentazione navigabile
```

### 2. Test su Macchina Pulita (Opzionale)

```bash
# Clone fresco
git clone https://github.com/atipiqal/roBERT.git
cd roBERT

# Segui QUICKSTART_GPT4MINI_WINDOWS.md
```

### 3. Comunica ai Collaboratori

Invia messaggio tipo:

```
📢 Repository roBERT aggiornata!

Principali novità:
- Documentazione completa CLI
- Server GPT-4o-mini funzionante
- Script PowerShell per Windows
- Quick start in 5 minuti

Leggi: RIEPILOGO_FINALE.md per dettagli completi
Inizia: QUICKSTART_GPT4MINI_WINDOWS.md
```

---

## 🆘 Troubleshooting

### "Ho committato file sbagliati"

```bash
# Annulla ultimo commit (mantiene modifiche)
git reset --soft HEAD~1

# Rimuovi file specifici dallo stage
git reset HEAD <file>

# Recommit
git add <file_corretti>
git commit -m "..."
```

### "Voglio modificare il messaggio"

```bash
# Se non hai ancora fatto push
git commit --amend

# Modifica messaggio nell'editor
# Salva e chiudi
```

### "Ho fatto push ma voglio cambiare qualcosa"

```bash
# ATTENZIONE: Usa solo se nessun altro ha fatto pull

# Modifica commit
git commit --amend
git push --force origin main

# MEGLIO: Fai un nuovo commit
git add <modifiche>
git commit -m "fix: correggi XYZ"
git push origin main
```

---

## ✅ Checklist Finale

Prima di `git push`:

- [ ] `robimb --help` funziona
- [ ] `git status` mostra solo file desiderati
- [ ] Nessun `.env` nello stage
- [ ] Messaggio commit descrive bene le modifiche
- [ ] File temporanei non inclusi (`.pyc`, `.log`, etc.)
- [ ] README link funzionano (verifica locale)

---

**Pronto per il commit! 🚀**

Per domande: vedi RIEPILOGO_FINALE.md
