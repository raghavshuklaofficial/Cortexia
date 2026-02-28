#!/bin/bash
# Cortexia — Health Monitor Script
# Checks application health and auto-restarts on failure.
#
# Add to crontab for 5-minute checks:
#   */5 * * * * /opt/cortexia/deploy/health-monitor.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/opt/cortexia}"
ENDPOINT="https://localhost/health"
LOG="/var/log/cortexia-health.log"
MAX_LOG_SIZE=10485760  # 10MB

# Rotate log if too large
if [ -f "$LOG" ] && [ "$(stat -f%z "$LOG" 2>/dev/null || stat -c%s "$LOG" 2>/dev/null || echo 0)" -gt "$MAX_LOG_SIZE" ]; then
    mv "$LOG" "$LOG.old"
fi

# Check health endpoint (skip cert validation for self-signed)
STATUS=$(curl -sk -o /dev/null -w "%{http_code}" --max-time 10 "$ENDPOINT" 2>/dev/null || echo "000")

if [ "$STATUS" = "200" ]; then
    # Healthy — only log every hour to reduce noise
    MINUTE=$(date +%M)
    if [ "$MINUTE" = "00" ]; then
        echo "$(date -u) OK: Health check passed (HTTP $STATUS)" >> "$LOG"
    fi
else
    echo "$(date -u) ALERT: Health check FAILED (HTTP $STATUS)" >> "$LOG"

    # Check which services are down
    cd "$PROJECT_DIR"
    UNHEALTHY=$(docker compose ps --format "{{.Name}} {{.Status}}" | grep -v "healthy\|running" || true)

    if [ -n "$UNHEALTHY" ]; then
        echo "$(date -u) Unhealthy services: $UNHEALTHY" >> "$LOG"
    fi

    # Attempt restart
    echo "$(date -u) Attempting restart..." >> "$LOG"
    docker compose -f "$PROJECT_DIR/docker-compose.yml" restart api
    sleep 30

    # Re-check
    STATUS2=$(curl -sk -o /dev/null -w "%{http_code}" --max-time 10 "$ENDPOINT" 2>/dev/null || echo "000")
    if [ "$STATUS2" = "200" ]; then
        echo "$(date -u) Recovery successful after restart (HTTP $STATUS2)" >> "$LOG"
    else
        echo "$(date -u) CRITICAL: Recovery FAILED (HTTP $STATUS2). Manual intervention required." >> "$LOG"
    fi
fi
