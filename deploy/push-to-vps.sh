#!/usr/bin/env bash
# =============================================================================
# Push code + DB dump to VPS
# =============================================================================
# Run from your LOCAL machine (inside nba-injury-scraper/):
#   bash deploy/push-to-vps.sh root@YOUR_DROPLET_IP
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash deploy/push-to-vps.sh user@host"
    echo "  e.g. bash deploy/push-to-vps.sh root@167.99.123.45"
    exit 1
fi

REMOTE="$1"
APP_DIR="/opt/nba-injury-scraper"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo " Pushing to $REMOTE"
echo "============================================"

# -------------------------------------------------------------------
# 1. Dump local MariaDB database
# -------------------------------------------------------------------
echo "[1/3] Dumping local database..."
DUMP_FILE="$PROJECT_DIR/nba_dump.sql"

# Try docker container first, fall back to local mysqldump
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "db\|sql\|maria"; then
    CONTAINER=$(docker ps --format '{{.Names}}' | grep -E 'db|sql|maria' | head -1)
    echo "  Using docker container: $CONTAINER"
    docker exec "$CONTAINER" mysqldump -unba_sql -pnba_sql --single-transaction nba > "$DUMP_FILE"
elif command -v mysqldump &>/dev/null; then
    echo "  Using local mysqldump"
    mysqldump -unba_sql -pnba_sql --single-transaction -h127.0.0.1 nba > "$DUMP_FILE"
else
    echo "  WARNING: Could not dump database. No docker container or mysqldump found."
    echo "  You'll need to transfer the database manually."
    DUMP_FILE=""
fi

if [ -n "$DUMP_FILE" ] && [ -f "$DUMP_FILE" ]; then
    DUMP_SIZE=$(du -h "$DUMP_FILE" | cut -f1)
    echo "  Dump size: $DUMP_SIZE"
fi

# -------------------------------------------------------------------
# 2. Rsync project files to VPS
# -------------------------------------------------------------------
echo "[2/3] Syncing project files..."
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

# -------------------------------------------------------------------
# 3. Transfer DB dump separately (can be large)
# -------------------------------------------------------------------
if [ -n "${DUMP_FILE:-}" ] && [ -f "$DUMP_FILE" ]; then
    echo "[3/3] Transferring database dump..."
    scp "$DUMP_FILE" "$REMOTE:$APP_DIR/nba_dump.sql"
    rm "$DUMP_FILE"
    echo "  Dump transferred and local copy removed."
else
    echo "[3/3] Skipping DB dump transfer."
fi

echo ""
echo "============================================"
echo " Done! Files are at $REMOTE:$APP_DIR"
echo "============================================"
echo ""
echo " SSH in and finish setup:"
echo "   ssh $REMOTE"
echo "   cd $APP_DIR"
echo ""
echo " Start services:"
echo "   docker compose up -d"
echo ""
echo " Restore database:"
echo "   docker exec -i \$(docker ps -qf name=db) mysql -unba_sql -p'PASSWORD' nba < nba_dump.sql"
echo ""
echo " Run migrations:"
echo "   conda run -n nba python main.py --migrate"
echo ""
echo " Test:"
echo "   curl http://localhost:8000/health"
echo ""
