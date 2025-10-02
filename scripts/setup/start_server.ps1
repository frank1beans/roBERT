# start_server.ps1
# Script PowerShell per avviare il server LLM GPT-4o-mini

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "     Avvio Server GPT-4o-mini per roBERT" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# Vai alla root del progetto (2 livelli sopra scripts/setup/)
$rootPath = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $rootPath
Write-Host "Directory: $rootPath" -ForegroundColor Gray
Write-Host ""

# Verifica API key
if (-not $env:OPENAI_API_KEY) {
    Write-Host "OPENAI_API_KEY non trovata nelle variabili d'ambiente" -ForegroundColor Yellow
    Write-Host "   Cerco nel file .env..." -ForegroundColor Yellow

    if (Test-Path .env) {
        Get-Content .env | ForEach-Object {
            if ($_ -match "^\s*OPENAI_API_KEY\s*=\s*(.+)$") {
                $apiKey = $matches[1].Trim().Trim('"').Trim("'")
                $env:OPENAI_API_KEY = $apiKey
                $keyPreview = $apiKey.Substring(0, [Math]::Min(15, $apiKey.Length)) + "..."
                Write-Host "   [OK] API Key caricata da .env: $keyPreview" -ForegroundColor Green
            }
        }
    } else {
        Write-Host ""
        Write-Host "[ERRORE] File .env non trovato!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Per configurare:" -ForegroundColor Yellow
        Write-Host "  1. Copia il template: Copy-Item .env.example .env" -ForegroundColor Gray
        Write-Host "  2. Modifica .env: notepad .env" -ForegroundColor Gray
        Write-Host "  3. Aggiungi: OPENAI_API_KEY=sk-proj-..." -ForegroundColor Gray
        Write-Host ""
        Write-Host "Oppure imposta variabile d'ambiente:" -ForegroundColor Yellow
        Write-Host '  $env:OPENAI_API_KEY = "sk-proj-..."' -ForegroundColor Gray
        Write-Host ""

        # Chiedi se continuare comunque
        $continue = Read-Host "Continuare comunque? (y/N)"
        if ($continue -ne "y" -and $continue -ne "Y") {
            exit 1
        }
    }
}

if ($env:OPENAI_API_KEY) {
    $keyPreview = $env:OPENAI_API_KEY.Substring(0, [Math]::Min(15, $env:OPENAI_API_KEY.Length)) + "..."
    Write-Host "[OK] OPENAI_API_KEY configurata: $keyPreview" -ForegroundColor Green
} else {
    Write-Host "[WARN] Server avviato SENZA API key - le richieste falliranno!" -ForegroundColor Yellow
}

Write-Host ""

# Verifica che il file del server esista
$serverPath = "examples/llm_server_gpt4mini.py"
if (-not (Test-Path $serverPath)) {
    Write-Host "[ERRORE] File server non trovato: $serverPath" -ForegroundColor Red
    exit 1
}

# Verifica che openai sia installato
Write-Host "Verifica dipendenze..." -ForegroundColor Gray
$openaiInstalled = pip list 2>$null | Select-String "openai"
if (-not $openaiInstalled) {
    Write-Host "[WARN] Modulo 'openai' non installato!" -ForegroundColor Yellow
    Write-Host "   Installazione in corso..." -ForegroundColor Yellow
    pip install openai
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERRORE] Errore durante l'installazione di openai" -ForegroundColor Red
        exit 1
    }
    Write-Host "   [OK] openai installato" -ForegroundColor Green
} else {
    Write-Host "   [OK] openai gia' installato" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "          Server in avvio sulla porta 8000" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Endpoint: http://localhost:8000/extract" -ForegroundColor Cyan
Write-Host "Health:   http://localhost:8000/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "Per fermare il server: CTRL+C" -ForegroundColor Yellow
Write-Host ""
Write-Host "--------------------------------------------------------" -ForegroundColor Gray
Write-Host ""

# Avvia server
python $serverPath --port 8000
