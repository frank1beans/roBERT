# 🚀 Quick Start GPT-4o-mini - Windows PowerShell

Guida rapida per usare GPT-4o-mini con roBERT su **Windows** in **5 minuti**.

## ⚡ Setup Veloce (PowerShell)

### 1. Installa dipendenza OpenAI

```powershell
pip install openai
```

### 2. Configura API Key

```powershell
# Ottieni API key da: https://platform.openai.com/api-keys

# Imposta variabile d'ambiente (sessione corrente)
$env:OPENAI_API_KEY = "sk-proj-TUA-KEY-QUI"

# Verifica
echo $env:OPENAI_API_KEY
```

**Per configurazione permanente**, vedi [sezione .env](#configurazione-permanente-env) sotto.

### 3. Avvia Server GPT-4o-mini

```powershell
python examples/llm_server_gpt4mini.py --port 8000
```

Dovresti vedere:
```
✅ OPENAI_API_KEY configured
🚀 Starting GPT-4o-mini LLM Server on http://0.0.0.0:8000
```

### 4. Test (in altro PowerShell)

**Apri un NUOVO terminale PowerShell** e testa:

```powershell
# Test connessione con curl (se disponibile)
curl http://localhost:8000/health

# Se curl non funziona, usa Invoke-WebRequest
Invoke-WebRequest -Uri "http://localhost:8000/health" -Method GET | Select-Object -ExpandProperty Content
```

Risposta attesa:
```json
{"status":"healthy","model":"gpt-4o-mini","api_key_configured":true}
```

### 5. Estrazione Proprietà

**Nel secondo terminale**, esegui:

```powershell
robimb extract properties `
  --input resources/data/train/classification/raw/dataset_lim.jsonl `
  --output outputs/extracted.jsonl `
  --llm-endpoint http://localhost:8000/extract `
  --llm-model gpt-4o-mini `
  --no-qa `
  --sample 10
```

**Nota**: In PowerShell usa `` ` `` (backtick) per continuare su più righe.

### 6. Analisi Risultati

```powershell
python scripts/analysis/extraction_results.py outputs/extracted.jsonl
```

## 🎯 Fatto!

Hai appena estratto proprietà usando GPT-4o-mini su Windows!

---

## 📝 Configurazione Permanente (.env)

### Metodo 1: File .env (Raccomandato)

```powershell
# 1. Copia template
Copy-Item .env.example .env

# 2. Apri con notepad
notepad .env

# 3. Modifica e aggiungi:
# OPENAI_API_KEY=sk-proj-TUA-KEY-QUI

# 4. Salva e chiudi
```

Il server caricherà automaticamente `.env` se hai installato `python-dotenv`:

```powershell
pip install python-dotenv
```

### Metodo 2: Variabile Permanente (Windows)

**PowerShell con privilegi Amministratore:**

```powershell
# Imposta variabile utente permanente
[System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'sk-proj-TUA-KEY', 'User')

# Verifica (riapri PowerShell dopo)
$env:OPENAI_API_KEY
```

**Oppure tramite GUI:**

1. `Windows + R` → `sysdm.cpl` → Invio
2. Tab "Avanzate" → "Variabili d'ambiente"
3. Sezione "Variabili utente" → "Nuova"
4. Nome: `OPENAI_API_KEY`
5. Valore: `sk-proj-TUA-KEY`
6. OK → OK

---

## 🔥 Workflow Windows

### Setup Iniziale (una volta)

```powershell
# 1. Installa dipendenze
pip install openai python-dotenv

# 2. Configura .env
Copy-Item .env.example .env
notepad .env
# Aggiungi: OPENAI_API_KEY=sk-proj-...
# Salva e chiudi

# 3. Verifica
python examples/llm_server_gpt4mini.py --port 8000
# CTRL+C per fermare
```

### Utilizzo Quotidiano

**Terminal 1 (Server):**
```powershell
cd C:\Users\f.biggi\Desktop\Script\roBERT
python examples/llm_server_gpt4mini.py --port 8000
```

**Terminal 2 (Estrazione):**
```powershell
cd C:\Users\f.biggi\Desktop\Script\roBERT

# Estrazione base
robimb extract properties `
  --input resources/data/train/classification/raw/dataset_lim.jsonl `
  --output outputs/estratto.jsonl `
  --llm-endpoint http://localhost:8000/extract `
  --llm-model gpt-4o-mini `
  --no-qa `
  --sample 20

# Analisi
python scripts/analysis/extraction_results.py outputs/estratto.jsonl
```

---

## 🛠️ Comandi PowerShell Utili

### Gestione Server

```powershell
# Avvia server
python examples/llm_server_gpt4mini.py --port 8000

# Avvia in background (con Start-Process)
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python examples/llm_server_gpt4mini.py --port 8000"

# Verifica se server è attivo
Test-NetConnection -ComputerName localhost -Port 8000

# Trova processo sulla porta 8000
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
Get-Process -Id <PID>

# Kill processo server
Stop-Process -Id <PID>
# Oppure CTRL+C nel terminale del server
```

### Test API

```powershell
# Health check (metodo 1 - curl)
curl http://localhost:8000/health

# Health check (metodo 2 - PowerShell nativo)
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET

# Test estrazione completo
$body = @{
    prompt = "Testo:`nPiastrella gres porcellanato 60x60 sp. 10mm`nDomanda:`nEstrai dimensioni`nSchema:`n{}"
    schema = @{}
    model = "gpt-4o-mini"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/extract" -Method POST -Body $body -ContentType "application/json"
```

### Gestione File

```powershell
# Verifica file esistenti
Test-Path examples/llm_server_gpt4mini.py
Test-Path .env

# Visualizza contenuto (senza API key!)
Get-Content .env.example

# Lista outputs
Get-ChildItem outputs/*.jsonl

# Apri file con editor
notepad outputs/extracted.jsonl

# Conta righe in JSONL
(Get-Content outputs/extracted.jsonl | Measure-Object -Line).Lines
```

---

## 💡 Tips Windows-Specific

### 1. Usa Windows Terminal

Migliore esperienza con tab multipli:
- Scarica da Microsoft Store: "Windows Terminal"
- Tab 1: Server LLM
- Tab 2: Comandi roBERT
- Tab 3: Analisi/monitoring

### 2. Path Lunghi

Se hai problemi con path troppo lunghi:

```powershell
# Abilita long paths (Admin PowerShell)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### 3. Encoding UTF-8

Se hai problemi con caratteri italiani:

```powershell
# All'inizio della sessione
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
```

### 4. Alias Utili

Aggiungi al tuo profilo PowerShell (`notepad $PROFILE`):

```powershell
# Alias per roBERT
function Start-LLMServer { python examples/llm_server_gpt4mini.py --port 8000 }
function Test-LLMServer { Invoke-RestMethod -Uri "http://localhost:8000/health" }
function Analyze-Extraction { param($file) python scripts/analysis/extraction_results.py $file }

# Usa con:
# Start-LLMServer
# Test-LLMServer
# Analyze-Extraction outputs/extracted.jsonl
```

---

## 🐛 Troubleshooting Windows

### "OPENAI_API_KEY not set"

```powershell
# Verifica
echo $env:OPENAI_API_KEY

# Se vuoto o null
$env:OPENAI_API_KEY = "sk-proj-..."

# Oppure usa .env (metodo raccomandato)
Copy-Item .env.example .env
notepad .env
# Aggiungi la key e salva
```

### "python: command not found"

```powershell
# Verifica installazione Python
python --version

# Se non funziona, prova
py --version
python3 --version

# Aggiungi Python al PATH o reinstalla da python.org
```

### "Module 'openai' not found"

```powershell
# Installa
pip install openai

# Se pip non funziona
python -m pip install openai

# Verifica installazione
pip list | Select-String openai
```

### "Port 8000 already in use"

```powershell
# Trova processo sulla porta 8000
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess

# Ottieni dettagli processo
$pid = (Get-NetTCPConnection -LocalPort 8000).OwningProcess
Get-Process -Id $pid

# Termina processo
Stop-Process -Id $pid

# Oppure usa porta diversa
python examples/llm_server_gpt4mini.py --port 8001
```

### "Access Denied" durante installazione

```powershell
# Esegui PowerShell come Amministratore
# Oppure installa per utente corrente
pip install --user openai
```

### Firewall blocca connessioni

```powershell
# Aggiungi regola firewall (PowerShell Admin)
New-NetFirewallRule -DisplayName "roBERT LLM Server" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

---

## 📊 Monitoring Risorse

### CPU e Memoria

```powershell
# Trova processo Python del server
Get-Process python | Where-Object {$_.MainWindowTitle -like "*llm*"} | Select-Object CPU, WorkingSet64, Id

# Monitor continuo (aggiorna ogni 2 secondi)
while($true) {
    cls
    Get-Process python | Format-Table Id, ProcessName, CPU, @{L='Memory(MB)';E={$_.WS/1MB -as [int]}}
    Start-Sleep -Seconds 2
}
# CTRL+C per fermare
```

---

## 📚 Script Batch Utili

### start_server.ps1

Crea file `start_server.ps1`:

```powershell
# start_server.ps1
Write-Host "🚀 Avvio Server GPT-4o-mini..." -ForegroundColor Green

# Verifica API key
if (-not $env:OPENAI_API_KEY) {
    Write-Host "⚠️  OPENAI_API_KEY non configurata!" -ForegroundColor Yellow
    Write-Host "Carico da .env..." -ForegroundColor Yellow

    if (Test-Path .env) {
        Get-Content .env | ForEach-Object {
            if ($_ -match "OPENAI_API_KEY=(.+)") {
                $env:OPENAI_API_KEY = $matches[1]
                Write-Host "✅ API Key caricata da .env" -ForegroundColor Green
            }
        }
    } else {
        Write-Host "❌ File .env non trovato!" -ForegroundColor Red
        exit 1
    }
}

# Avvia server
python examples/llm_server_gpt4mini.py --port 8000
```

Usa con:
```powershell
.\start_server.ps1
```

### extract.ps1

Crea file `extract.ps1`:

```powershell
# extract.ps1
param(
    [int]$Sample = 10,
    [string]$Input = "resources/data/train/classification/raw/dataset_lim.jsonl",
    [string]$Output = "outputs/extracted.jsonl"
)

Write-Host "📝 Estrazione con GPT-4o-mini" -ForegroundColor Cyan
Write-Host "   Sample: $Sample" -ForegroundColor Gray
Write-Host "   Input: $Input" -ForegroundColor Gray
Write-Host "   Output: $Output" -ForegroundColor Gray

robimb extract properties `
    --input $Input `
    --output $Output `
    --llm-endpoint http://localhost:8000/extract `
    --llm-model gpt-4o-mini `
    --no-qa `
    --sample $Sample

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Estrazione completata!" -ForegroundColor Green
    Write-Host "📊 Analisi risultati:" -ForegroundColor Cyan
    python scripts/analysis/extraction_results.py $Output
} else {
    Write-Host "❌ Errore durante l'estrazione" -ForegroundColor Red
}
```

Usa con:
```powershell
.\extract.ps1 -Sample 20
.\extract.ps1 -Sample 50 -Output "outputs/test.jsonl"
```

---

## 🎓 Prossimi Passi

1. ✅ **Setup completato** - Server funzionante!
2. 📖 **Leggi documentazione**: [examples/README.md](examples/README.md)
3. 🧪 **Sperimenta** con sample diversi (10, 50, 100)
4. 📊 **Confronta** risultati: mock vs GPT-4o-mini vs rules-only
5. 💰 **Monitora** costi su: https://platform.openai.com/usage

---

## 📞 Supporto

- **Documentazione completa**: [examples/README.md](examples/README.md)
- **Comandi CLI**: [docs/commands/extract.md](docs/commands/extract.md)
- **Architettura**: [ARCHITECTURE.md](ARCHITECTURE.md)

---

**Buona estrazione su Windows! 🎉**
