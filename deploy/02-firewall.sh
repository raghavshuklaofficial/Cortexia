#!/bin/bash
# Cortexia — UFW Firewall Setup
# Blocks all incoming traffic except SSH, HTTP, and HTTPS.
set -euo pipefail

SSH_PORT="${1:-2222}"

echo "=== Firewall Setup (UFW) ==="

apt-get update -qq && apt-get install -y -qq ufw

# Reset to defaults
ufw --force reset

# Default policies: deny incoming, allow outgoing
ufw default deny incoming
ufw default allow outgoing

# Allow essential ports only
ufw allow "$SSH_PORT"/tcp comment "SSH"
ufw allow 80/tcp comment "HTTP (redirect to HTTPS)"
ufw allow 443/tcp comment "HTTPS"

# Block common botnet attack ports
ufw deny 23/tcp comment "Block Telnet"
ufw deny 2323/tcp comment "Block Telnet alt"
ufw deny 7547/tcp comment "Block TR-069 (ISP management)"
ufw deny 5555/tcp comment "Block ADB (Android Debug)"
ufw deny 37215/tcp comment "Block Huawei router exploit"

# Enable firewall
ufw --force enable

echo ""
echo "Firewall enabled. Status:"
ufw status verbose
