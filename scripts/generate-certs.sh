#!/bin/bash
# Generate self-signed TLS certificate for Cortexia (raw IP deployment).
# Usage: bash scripts/generate-certs.sh <VPS_IP_ADDRESS>
set -euo pipefail

CERT_DIR="$(cd "$(dirname "$0")/.." && pwd)/docker/certs"
mkdir -p "$CERT_DIR"

VPS_IP="${1:?Usage: $0 <VPS_IP_ADDRESS>}"

echo "Generating self-signed TLS certificate for IP: $VPS_IP"

openssl req -x509 -nodes -days 365 \
    -newkey rsa:2048 \
    -keyout "$CERT_DIR/cortexia.key" \
    -out "$CERT_DIR/cortexia.crt" \
    -subj "/CN=$VPS_IP/O=Cortexia/C=US" \
    -addext "subjectAltName=IP:$VPS_IP"

chmod 600 "$CERT_DIR/cortexia.key"
chmod 644 "$CERT_DIR/cortexia.crt"

echo ""
echo "Certificates generated:"
echo "  Certificate: $CERT_DIR/cortexia.crt"
echo "  Private key: $CERT_DIR/cortexia.key"
echo ""
echo "Note: Browsers will show a security warning for self-signed certificates."
echo "This is expected for raw-IP deployments."
echo ""
echo "To upgrade to Let's Encrypt (requires a domain name):"
echo "  1. apt install certbot python3-certbot-nginx"
echo "  2. certbot certonly --standalone -d yourdomain.com"
echo "  3. Update docker-compose.yml cert volume to /etc/letsencrypt/live/yourdomain.com/"
echo "  4. Add certbot renewal cron: 0 0 1 * * certbot renew --quiet"
