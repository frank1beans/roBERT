# Setup GPT-4o-mini per roBERT
# Questo script automatizza la configurazione e il test di GPT-4o-mini

param(
    [string]$ApiKey = "",
    [switch]$SkipTest,
    [int]$TestSample = 20
)

$ErrorActionPreference = "Stop"

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "GPT-4o-mini Setup per roBERT" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python
Write-Host "[1/6] Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  ✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Python not found!" -ForegroundColor Red
    Write-Host "  Please install Python 3.8+ from https://www.python.org/" -ForegroundColor Red
    exit 1
}

# Step 2: Install openai package
Write-Host "`n[2/6] Installing/Checking openai package..." -ForegroundColor Yellow
$openaiCheck = pip show openai 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ openai package already installed" -ForegroundColor Green
} else {
    Write-Host "  Installing openai..." -ForegroundColor Yellow
    pip install openai
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ openai package installed successfully" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Failed to install openai package" -ForegroundColor Red
        exit 1
    }
}

# Step 3: Get API Key
Write-Host "`n[3/6] Configuring API Key..." -ForegroundColor Yellow
if ($ApiKey -eq "") {
    # Check environment variable
    $envKey = $env:OPENAI_API_KEY
    if ($envKey) {
        Write-Host "  ✓ Using API key from environment variable" -ForegroundColor Green
        $ApiKey = $envKey
    } else {
        # Prompt user
        Write-Host "  Enter your OpenAI API Key (starts with sk-...):" -ForegroundColor White
        $ApiKey = Read-Host "  API Key"
        if ($ApiKey -eq "") {
            Write-Host "  ✗ No API key provided" -ForegroundColor Red
            Write-Host "  Get one from: https://platform.openai.com/api-keys" -ForegroundColor Yellow
            exit 1
        }
    }
}

# Set environment variable
$env:OPENAI_API_KEY = $ApiKey
Write-Host "  ✓ API key configured" -ForegroundColor Green

# Step 4: Verify API Key
Write-Host "`n[4/6] Verifying API Key..." -ForegroundColor Yellow
$testScript = @'
import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
response = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "test"}],
    max_tokens=5
)
print("OK")
'@

$result = $testScript | python 2>&1
if ($result -match "OK") {
    Write-Host "  ✓ API key verified successfully" -ForegroundColor Green
} else {
    Write-Host "  ✗ API key verification failed" -ForegroundColor Red
    Write-Host "  Error: $result" -ForegroundColor Red
    Write-Host "`n  Please check:" -ForegroundColor Yellow
    Write-Host "  1. Your API key is correct" -ForegroundColor Yellow
    Write-Host "  2. You have credits in your OpenAI account" -ForegroundColor Yellow
    Write-Host "  3. You have access to gpt-4o-mini model" -ForegroundColor Yellow
    exit 1
}

# Step 5: Start server in background
Write-Host "`n[5/6] Starting GPT-4o-mini server..." -ForegroundColor Yellow
$serverJob = Start-Job -ScriptBlock {
    param($apiKey)
    $env:OPENAI_API_KEY = $apiKey
    Set-Location $using:PWD
    python llm_server_gpt4mini.py --port 8000
} -ArgumentList $ApiKey

# Wait for server to start
Start-Sleep -Seconds 3

# Check if server is running
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
    if ($health.status -eq "healthy") {
        Write-Host "  ✓ Server started successfully on http://localhost:8000" -ForegroundColor Green
    }
} catch {
    Write-Host "  ✗ Server failed to start" -ForegroundColor Red
    Stop-Job $serverJob
    Remove-Job $serverJob
    exit 1
}

# Step 6: Run test extraction (if not skipped)
if (-not $SkipTest) {
    $testMsg = "`n[6/6] Running test extraction with $TestSample samples..."
    Write-Host $testMsg -ForegroundColor Yellow
    Write-Host "  This may take 1-2 minutes..." -ForegroundColor Gray

    robimb extract properties `
        --input resources\data\train\classification\raw\dataset_lim.jsonl `
        --output outputs\gpt4mini_setup_test.jsonl `
        --llm-endpoint http://localhost:8000/extract `
        --llm-model gpt-4o-mini `
        --no-qa `
        --max-workers 8 `
        --sample $TestSample 2>&1 | Out-Null

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Test extraction completed successfully" -ForegroundColor Green

        # Analyze results if script exists
        if (Test-Path "analyze_extraction.py") {
            Write-Host "`n  Analyzing results..." -ForegroundColor Gray
            python analyze_extraction.py outputs\gpt4mini_setup_test.jsonl
        }
    } else {
        Write-Host "  ⚠ Test extraction had issues" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n[6/6] Skipping test extraction" -ForegroundColor Yellow
}

# Step 7: Show stats
Write-Host "`n" + "=" * 80 -ForegroundColor Cyan
Write-Host "Setup Completed Successfully!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Server Information:" -ForegroundColor White
Write-Host "  Status:     Running (Job ID: $($serverJob.Id))" -ForegroundColor White
Write-Host "  Endpoint:   http://localhost:8000/extract" -ForegroundColor White
Write-Host "  Stats:      http://localhost:8000/stats" -ForegroundColor White
Write-Host "  Model:      gpt-4o-mini" -ForegroundColor White
Write-Host ""

# Get stats
try {
    $stats = Invoke-RestMethod -Uri "http://localhost:8000/stats"
    if ($stats.total_requests -gt 0) {
        Write-Host "Current Statistics:" -ForegroundColor White
        Write-Host "  Requests:   $($stats.total_requests) ($($stats.successful_requests) successful)" -ForegroundColor White
        Write-Host "  Tokens:     $($stats.total_input_tokens + $stats.total_output_tokens) total" -ForegroundColor White
        Write-Host "  Cost:       `$$([math]::Round($stats.total_cost, 4))" -ForegroundColor White
        Write-Host ""
    }
} catch {
    # Ignore if stats not available
}

Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Test extraction:" -ForegroundColor White
Write-Host "     robimb extract properties --input data.jsonl --output results.jsonl --llm-endpoint http://localhost:8000/extract --llm-model gpt-4o-mini --max-workers 8 --sample 50" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. View stats:" -ForegroundColor White
Write-Host "     Invoke-RestMethod -Uri http://localhost:8000/stats" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Stop server:" -ForegroundColor White
Write-Host "     Stop-Job $($serverJob.Id); Remove-Job $($serverJob.Id)" -ForegroundColor Gray
Write-Host ""
Write-Host "Documentation: SETUP_GPT4MINI.md" -ForegroundColor Yellow
Write-Host ""

# Keep server running
Write-Host "Press Ctrl+C to stop the server and exit" -ForegroundColor Cyan
try {
    Wait-Job $serverJob
} finally {
    Stop-Job $serverJob -ErrorAction SilentlyContinue
    Remove-Job $serverJob -ErrorAction SilentlyContinue
}
