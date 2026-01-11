# =============================================================================
# Cloud Inference Script for DeepResearch (PowerShell)
# This script runs inference using cloud APIs (OpenRouter, OpenAI, etc.)
# No local vLLM servers required
# =============================================================================

$ErrorActionPreference = "Stop"

# Load environment variables from .env file
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvFile = Join-Path (Split-Path -Parent $ScriptDir) ".env"

if (-not (Test-Path $EnvFile)) {
    Write-Host "Error: .env file not found at $EnvFile" -ForegroundColor Red
    Write-Host "Please copy .env.example to .env and configure your settings:"
    Write-Host "  cp .env.example .env"
    exit 1
}

Write-Host "Loading environment variables from .env file..."

# Parse .env file and set environment variables
Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#")) {
        $parts = $line -split "=", 2
        if ($parts.Count -eq 2) {
            $key = $parts[0].Trim()
            $value = $parts[1].Trim()
            # Remove surrounding quotes if present
            $value = $value -replace '^["'']|["'']$', ''
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

# Get environment variables
$MainModelApiKey = $env:MAIN_MODEL_API_KEY
$MainModelBaseUrl = $env:MAIN_MODEL_BASE_URL
$MainModelName = $env:MAIN_MODEL_NAME
$Tokenizer = $env:TOKENIZER
$Dataset = $env:DATASET
$OutputPath = $env:OUTPUT_PATH
$MaxWorkers = if ($env:MAX_WORKERS) { $env:MAX_WORKERS } else { "20" }
$Temperature = if ($env:TEMPERATURE) { $env:TEMPERATURE } else { "0.6" }
$PresencePenalty = if ($env:PRESENCE_PENALTY) { $env:PRESENCE_PENALTY } else { "1.1" }
$RolloutCount = if ($env:ROLLOUT_COUNT) { $env:ROLLOUT_COUNT } else { "3" }

# Validate cloud API configuration
if (-not $MainModelApiKey -or $MainModelApiKey -eq "your_api_key") {
    Write-Host "Error: MAIN_MODEL_API_KEY not configured in .env file" -ForegroundColor Red
    exit 1
}

if (-not $MainModelBaseUrl) {
    Write-Host "Error: MAIN_MODEL_BASE_URL not configured in .env file" -ForegroundColor Red
    exit 1
}

if (-not $MainModelName) {
    Write-Host "Error: MAIN_MODEL_NAME not configured in .env file" -ForegroundColor Red
    exit 1
}

if (-not $Tokenizer -or $Tokenizer -eq "your_tokenizer_model_id") {
    Write-Host "Error: TOKENIZER not configured in .env file" -ForegroundColor Red
    exit 1
}

Write-Host "========================================"
Write-Host "Cloud Inference Configuration:"
Write-Host "  API Base URL: $MainModelBaseUrl"
Write-Host "  Model: $MainModelName"
Write-Host "  Tokenizer: $Tokenizer"
Write-Host "  Dataset: $Dataset"
Write-Host "  Output Path: $OutputPath"
Write-Host "  Max Workers: $MaxWorkers"
Write-Host "  Temperature: $Temperature"
Write-Host "  Rollout Count: $RolloutCount"
Write-Host "========================================"

Set-Location $ScriptDir

python -u run_multi_react.py `
    --dataset "$Dataset" `
    --output "$OutputPath" `
    --max_workers $MaxWorkers `
    --model "$MainModelName" `
    --temperature $Temperature `
    --presence_penalty $PresencePenalty `
    --roll_out_count $RolloutCount
