#!/bin/bash

# =============================================================================
# Cloud Inference Script for DeepResearch
# This script runs inference using cloud APIs (OpenRouter, OpenAI, etc.)
# No local vLLM servers required
# =============================================================================

# Load environment variables from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please copy .env.example to .env and configure your settings:"
    echo "  cp .env.example .env"
    exit 1
fi

echo "Loading environment variables from .env file..."
set -a  # automatically export all variables
source "$ENV_FILE"
set +a  # stop automatically exporting

# Validate cloud API configuration
if [ -z "$MAIN_MODEL_API_KEY" ] || [ "$MAIN_MODEL_API_KEY" = "your_api_key" ]; then
    echo "Error: MAIN_MODEL_API_KEY not configured in .env file"
    exit 1
fi

if [ -z "$MAIN_MODEL_BASE_URL" ]; then
    echo "Error: MAIN_MODEL_BASE_URL not configured in .env file"
    exit 1
fi

if [ -z "$MAIN_MODEL_NAME" ]; then
    echo "Error: MAIN_MODEL_NAME not configured in .env file"
    exit 1
fi

if [ -z "$TOKENIZER" ] || [ "$TOKENIZER" = "your_tokenizer_model_id" ]; then
    echo "Error: TOKENIZER not configured in .env file"
    exit 1
fi

echo "========================================"
echo "Cloud Inference Configuration:"
echo "  API Base URL: $MAIN_MODEL_BASE_URL"
echo "  Model: $MAIN_MODEL_NAME"
echo "  Tokenizer: $TOKENIZER"
echo "  Dataset: $DATASET"
echo "  Output Path: $OUTPUT_PATH"
echo "  Max Workers: ${MAX_WORKERS:-20}"
echo "  Temperature: ${TEMPERATURE:-0.6}"
echo "  Rollout Count: ${ROLLOUT_COUNT:-3}"
echo "========================================"

cd "$SCRIPT_DIR"

python -u run_multi_react.py \
    --dataset "$DATASET" \
    --output "$OUTPUT_PATH" \
    --max_workers ${MAX_WORKERS:-20} \
    --model "${MAIN_MODEL_NAME}" \
    --temperature ${TEMPERATURE:-0.6} \
    --presence_penalty ${PRESENCE_PENALTY:-1.1} \
    --roll_out_count ${ROLLOUT_COUNT:-3}
