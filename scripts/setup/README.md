# Setup Scripts

Script di setup e configurazione per roBERT.

## 📋 Script Disponibili

### Windows PowerShell

#### start_server.ps1

Avvia il server GPT-4o-mini con verifica automatica dipendenze e API key.

**Utilizzo:**
```powershell
.\scripts\setup\start_server.ps1
```

**Funzionalità:**
- ✅ Verifica e carica `OPENAI_API_KEY` da `.env` se disponibile
- ✅ Controlla installazione modulo `openai`
- ✅ Installa dipendenze mancanti automaticamente
- ✅ Avvia server sulla porta 8000
- ✅ Mostra endpoint e istruzioni

#### extract.ps1

Script avanzato per estrazione proprietà con GPT-4o-mini.

**Utilizzo base:**
```powershell
# Estrazione con default (10 documenti)
.\scripts\setup\extract.ps1

# Estrazione 50 documenti
.\scripts\setup\extract.ps1 -Sample 50

# Estrazione con analisi automatica
.\scripts\setup\extract.ps1 -Sample 20 -Analyze

# Estrazione custom
.\scripts\setup\extract.ps1 `
    -Sample 100 `
    -InputFile "data/mio_dataset.jsonl" `
    -OutputFile "outputs/risultati.jsonl" `
    -Analyze
```

**Parametri:**
- `-Sample` : Numero documenti da processare (default: 10)
- `-InputFile` : File JSONL input (default: dataset_lim.jsonl)
- `-OutputFile` : File JSONL output (default: outputs/extracted.jsonl)
- `-Endpoint` : URL endpoint LLM (default: http://localhost:8000/extract)
- `-Model` : Modello LLM (default: gpt-4o-mini)
- `-Analyze` : Flag per analisi automatica risultati

**Funzionalità:**
- ✅ Verifica connessione server LLM
- ✅ Valida file di input
- ✅ Mostra configurazione prima dell'esecuzione
- ✅ Misura tempo di esecuzione
- ✅ Conta documenti elaborati
- ✅ Stima costi API
- ✅ Analisi automatica opzionale

### Linux/Mac

#### setup_gpt4mini.ps1

Script PowerShell originale (legacy) - usa `start_server.ps1` invece.

---

## 🚀 Workflow Completo Windows

### Prima Volta (Setup)

```powershell
# 1. Installa dipendenze
pip install openai python-dotenv

# 2. Configura API key
Copy-Item .env.example .env
notepad .env
# Aggiungi: OPENAI_API_KEY=sk-proj-...
# Salva e chiudi

# 3. Test server
.\scripts\setup\start_server.ps1
# CTRL+C dopo verifica

# 4. Test estrazione
.\scripts\setup\extract.ps1 -Sample 5 -Analyze
```

### Uso Quotidiano

**Terminal 1 (Server):**
```powershell
.\scripts\setup\start_server.ps1
# Lascia in esecuzione
```

**Terminal 2 (Estrazioni):**
```powershell
# Estrazione rapida
.\scripts\setup\extract.ps1 -Sample 20 -Analyze

# Batch grande
.\scripts\setup\extract.ps1 -Sample 500 -OutputFile "outputs/batch_$(Get-Date -Format 'yyyyMMdd').jsonl"
```

---

## 📝 Esempi Avanzati

### Pipeline Completa

```powershell
# Terminal 1: Avvia server
.\scripts\setup\start_server.ps1

# Terminal 2: Workflow
# 1. Estrazione piccola (test)
.\scripts\setup\extract.ps1 -Sample 10 -OutputFile "outputs/test.jsonl" -Analyze

# 2. Se OK, batch medio
.\scripts\setup\extract.ps1 -Sample 100 -OutputFile "outputs/medio.jsonl" -Analyze

# 3. Batch completo
.\scripts\setup\extract.ps1 -Sample 1000 -OutputFile "outputs/completo.jsonl" -Analyze
```

### Confronto Mock vs GPT-4o-mini

```powershell
# Terminal 1: Server GPT-4o-mini (porta 8000)
.\scripts\setup\start_server.ps1

# Terminal 2: Server Mock (porta 8001)
python examples/llm_server_example.py --port 8001

# Terminal 3: Estrazione con GPT-4o-mini
.\scripts\setup\extract.ps1 `
    -Sample 50 `
    -Endpoint "http://localhost:8000/extract" `
    -OutputFile "outputs/gpt4mini.jsonl" `
    -Analyze

# Terminal 3: Estrazione con Mock
.\scripts\setup\extract.ps1 `
    -Sample 50 `
    -Endpoint "http://localhost:8001/extract" `
    -OutputFile "outputs/mock.jsonl" `
    -Analyze

# Confronta risultati
python scripts/analysis/extraction_results.py outputs/gpt4mini.jsonl
python scripts/analysis/extraction_results.py outputs/mock.jsonl
```

### Batch Processing con Logging

```powershell
# Crea directory log
New-Item -ItemType Directory -Force -Path logs

# Estrazione con log dettagliato
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
.\scripts\setup\extract.ps1 `
    -Sample 500 `
    -OutputFile "outputs/batch_$timestamp.jsonl" `
    -Analyze `
    *>&1 | Tee-Object -FilePath "logs/extraction_$timestamp.log"
```

---

## 🔧 Troubleshooting

### Script non eseguibile

```powershell
# Verifica policy esecuzione
Get-ExecutionPolicy

# Se Restricted, cambia in RemoteSigned (PowerShell Admin)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Path non trovati

```powershell
# Esegui dalla root del progetto
cd C:\Users\f.biggi\Desktop\Script\roBERT
.\scripts\setup\start_server.ps1
```

### API Key non caricata

```powershell
# Verifica .env
Get-Content .env | Select-String "OPENAI_API_KEY"

# Carica manualmente
$env:OPENAI_API_KEY = "sk-proj-..."
.\scripts\setup\start_server.ps1
```

---

## 📚 Vedi Anche

- [QUICKSTART_GPT4MINI_WINDOWS.md](../../QUICKSTART_GPT4MINI_WINDOWS.md) - Guida completa Windows
- [examples/README.md](../../examples/README.md) - Documentazione server LLM
- [docs/commands/extract.md](../../docs/commands/extract.md) - Comando extract dettagliato
