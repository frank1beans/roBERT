# extract.ps1
# Script PowerShell per estrarre proprieta' con GPT-4o-mini

param(
    [Parameter(HelpMessage="Numero di documenti da processare")]
    [int]$Sample = 300,

    [Parameter(HelpMessage="File JSONL di input")]
    [string]$InputFile = "resources/data/train/classification/raw/dataset_lim.jsonl",

    [Parameter(HelpMessage="File JSONL di output")]
    [string]$OutputFile = "outputs/extracted.jsonl",

    [Parameter(HelpMessage="Endpoint LLM")]
    [string]$Endpoint = "http://localhost:8000/extract",

    [Parameter(HelpMessage="Modello LLM")]
    [string]$Model = "gpt-4o-mini",

    [Parameter(HelpMessage="Esegui analisi dopo estrazione")]
    [switch]$Analyze
)

# Vai alla root del progetto
$rootPath = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $rootPath

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "       Estrazione Proprieta' con LLM" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# Mostra configurazione
Write-Host "Configurazione:" -ForegroundColor Yellow
Write-Host "   Sample:   $Sample documenti" -ForegroundColor Gray
Write-Host "   Input:    $InputFile" -ForegroundColor Gray
Write-Host "   Output:   $OutputFile" -ForegroundColor Gray
Write-Host "   Endpoint: $Endpoint" -ForegroundColor Gray
Write-Host "   Model:    $Model" -ForegroundColor Gray
Write-Host ""

# Verifica che il file di input esista
if (-not (Test-Path $InputFile)) {
    Write-Host "[ERRORE] File di input non trovato: $InputFile" -ForegroundColor Red
    exit 1
}

# Verifica connessione al server LLM
Write-Host "Verifica server LLM..." -ForegroundColor Gray
try {
    $healthUrl = $Endpoint -replace '/extract$', '/health'
    $health = Invoke-RestMethod -Uri $healthUrl -Method GET -ErrorAction Stop

    if ($health.status -eq "healthy") {
        Write-Host "   [OK] Server LLM raggiungibile" -ForegroundColor Green
        Write-Host "   Model: $($health.model)" -ForegroundColor Gray
        Write-Host "   API Key: $($health.api_key_configured)" -ForegroundColor Gray
    } else {
        Write-Host "   [WARN] Server risponde ma status non healthy: $($health.status)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   [ERRORE] Server LLM non raggiungibile su $healthUrl" -ForegroundColor Red
    Write-Host "   Avvia prima il server con: .\scripts\setup\start_server.ps1" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continuare comunque? (y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        exit 1
    }
}

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "            Estrazione in corso..." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host ""

# Esegui estrazione
$startTime = Get-Date

robimb extract properties `
    --input $InputFile `
    --output $OutputFile `
    --llm-endpoint $Endpoint `
    --llm-model $Model `
    --no-qa `
    --sample $Sample

$exitCode = $LASTEXITCODE
$endTime = Get-Date
$duration = ($endTime - $startTime).TotalSeconds

Write-Host ""
Write-Host "--------------------------------------------------------" -ForegroundColor Gray
Write-Host ""

if ($exitCode -eq 0) {
    Write-Host "[OK] Estrazione completata in $([Math]::Round($duration, 2)) secondi!" -ForegroundColor Green

    # Conta righe estratte
    if (Test-Path $OutputFile) {
        $lines = (Get-Content $OutputFile | Measure-Object -Line).Lines
        Write-Host "Documenti elaborati: $lines" -ForegroundColor Cyan

        # Calcola stima costo (approssimativa)
        $estimatedCost = [Math]::Round($lines * 0.00006, 4)
        Write-Host "Costo stimato: ~`$$estimatedCost" -ForegroundColor Yellow
    }

    # Analisi automatica se richiesta
    if ($Analyze) {
        Write-Host ""
        Write-Host "Analisi risultati:" -ForegroundColor Cyan
        Write-Host ""
        python scripts/analysis/extraction_results.py $OutputFile
    } else {
        Write-Host ""
        Write-Host "Per analizzare i risultati:" -ForegroundColor Yellow
        Write-Host "   python scripts/analysis/extraction_results.py $OutputFile" -ForegroundColor Gray
    }
} else {
    Write-Host "[ERRORE] Errore durante l'estrazione (codice: $exitCode)" -ForegroundColor Red
    exit $exitCode
}

Write-Host ""
