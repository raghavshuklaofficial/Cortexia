#!/bin/bash
# Cortexia — Automated Backup Script
# Backs up PostgreSQL and Redis data with 30-day retention.
#
# Add to crontab for daily backups at 2 AM:
#   0 2 * * * /opt/cortexia/deploy/backup.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/opt/cortexia}"
BACKUP_DIR="$PROJECT_DIR/backups"
RETENTION_DAYS=30
DATE=$(date +%Y-%m-%d_%H%M)
LOG="/var/log/cortexia-backup.log"

# Source .env for cron context (provides POSTGRES_USER, POSTGRES_DB, etc.)
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    . "$PROJECT_DIR/.env"
    set +a
fi

mkdir -p "$BACKUP_DIR"

echo "$(date -u) Starting backup..." >> "$LOG"

# PostgreSQL dump via Docker
if docker compose -f "$PROJECT_DIR/docker-compose.yml" ps --status running postgres 2>/dev/null | grep -q postgres; then
    docker compose -f "$PROJECT_DIR/docker-compose.yml" exec -T postgres \
        pg_dump -U "${POSTGRES_USER:-cortexia}" "${POSTGRES_DB:-cortexia}" \
        | gzip > "$BACKUP_DIR/cortexia-db-$DATE.sql.gz"
    echo "$(date -u) PostgreSQL backup: cortexia-db-$DATE.sql.gz" >> "$LOG"
else
    echo "$(date -u) WARNING: PostgreSQL not running, skipping DB backup" >> "$LOG"
fi

# Redis data backup via temporary container
if docker compose -f "$PROJECT_DIR/docker-compose.yml" ps --status running redis 2>/dev/null | grep -q redis; then
    docker run --rm \
        -v "${PROJECT_DIR##*/}_redisdata:/data:ro" \
        -v "$BACKUP_DIR":/backup \
        alpine tar czf "/backup/redis-data-$DATE.tar.gz" -C /data .
    echo "$(date -u) Redis backup: redis-data-$DATE.tar.gz" >> "$LOG"
else
    echo "$(date -u) WARNING: Redis not running, skipping Redis backup" >> "$LOG"
fi

# Cleanup old backups
DELETED=$(find "$BACKUP_DIR" -name "*.gz" -mtime +"$RETENTION_DAYS" -delete -print | wc -l)
if [ "$DELETED" -gt 0 ]; then
    echo "$(date -u) Cleaned up $DELETED old backup(s)" >> "$LOG"
fi

echo "$(date -u) Backup completed successfully" >> "$LOG"
