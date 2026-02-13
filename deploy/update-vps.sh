#!/usr/bin/env bash
# =============================================================================
# Update VPS with latest code (no DB dump, just code changes)
# =============================================================================
# Run from your LOCAL machine:
#   bash deploy/update-vps.sh root@YOUR_DROPLET_IP
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash deploy/update-vps.sh user@host"
    exit 1
fi

REMOTE="$1"
APP_DIR="/opt/nba-injury-scraper"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Syncing code to $REMOTE..."
rsync -avz --progress \
    --exclude '.env' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.venv' \
    --exclude 'venv' \
    --exclude 'node_modules' \
    --exclude '.next' \
    --exclude 'data/*.db' \
    --exclude '.DS_Store' \
    --exclude 'dataset_cache.pkl' \
    --exclude 'nba_dump.sql' \
    "$PROJECT_DIR/" "$REMOTE:$APP_DIR/"

echo "Restarting API container..."
ssh "$REMOTE" "cd $APP_DIR && docker compose up -d --build api"

echo "Done. API restarted with latest code."
