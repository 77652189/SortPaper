$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$env:QWEN3_EMBEDDING_MODEL = if ($env:QWEN3_EMBEDDING_MODEL) {
    $env:QWEN3_EMBEDDING_MODEL
} else {
    "data/models/Qwen3-Embedding-0.6B"
}
$env:QWEN3_EMBEDDING_DEVICE = if ($env:QWEN3_EMBEDDING_DEVICE) {
    $env:QWEN3_EMBEDDING_DEVICE
} else {
    "cpu"
}
$env:QWEN3_EMBEDDING_PORT = if ($env:QWEN3_EMBEDDING_PORT) {
    $env:QWEN3_EMBEDDING_PORT
} else {
    "8001"
}

& .\.venv\Scripts\python.exe tools\qwen3_embedding_server.py
