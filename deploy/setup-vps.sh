#!/bin/bash
# Cortexia — Master VPS Setup Script
# Run as root on a fresh VPS (Ubuntu/Debian).
#
# PREREQUISITE: Your SSH public key must already be installed
# in ~/.ssh/authorized_keys for the non-root user.
#
# Usage: sudo bash deploy/setup-vps.sh
set -euo pipefail

echo "╔══════════════════════════════════════════════════╗"
echo "║     CORTEXIA — VPS Security Hardening Setup      ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "This script will:"
echo "  1. Harden SSH (key-only, custom port)"
echo "  2. Configure UFW firewall"
echo "  3. Install Fail2ban"
echo "  4. Install CrowdSec community IPS"
echo "  5. Enable automatic security updates"
echo "  6. Harden kernel networking"
echo "  7. Install Docker"
echo ""

# Check root
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script must be run as root."
    exit 1
fi

SSH_PORT="${SSH_PORT:-2222}"
read -rp "SSH port [$SSH_PORT]: " input_port
SSH_PORT="${input_port:-$SSH_PORT}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "=== Step 1/7: SSH Hardening ==="
bash "$SCRIPT_DIR/01-ssh-harden.sh" "$SSH_PORT"

echo ""
echo "=== Step 2/7: Firewall ==="
bash "$SCRIPT_DIR/02-firewall.sh" "$SSH_PORT"

echo ""
echo "=== Step 3/7: Fail2ban ==="
bash "$SCRIPT_DIR/03-fail2ban.sh" "$SSH_PORT"

echo ""
echo "=== Step 4/7: CrowdSec ==="
bash "$SCRIPT_DIR/04-crowdsec.sh"

echo ""
echo "=== Step 5/7: Auto Updates ==="
bash "$SCRIPT_DIR/05-auto-updates.sh"

echo ""
echo "=== Step 6/7: Kernel Hardening ==="
bash "$SCRIPT_DIR/06-kernel-harden.sh"

echo ""
echo "=== Step 7/7: Docker ==="
if command -v docker &> /dev/null; then
    echo "Docker already installed: $(docker --version)"
else
    curl -fsSL https://get.docker.com | sh
    # Add the invoking user to docker group if running via sudo
    if [ -n "${SUDO_USER:-}" ]; then
        usermod -aG docker "$SUDO_USER"
        echo "Added $SUDO_USER to docker group."
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║          VPS HARDENING COMPLETE                  ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  SSH port:      $SSH_PORT"
echo "║  Firewall:      UFW (80, 443, $SSH_PORT)"
echo "║  IPS:           Fail2ban + CrowdSec"
echo "║  Auto-updates:  Enabled"
echo "║  Kernel:        Hardened"
echo "║  Docker:        Installed"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Test SSH in a NEW terminal: ssh -p $SSH_PORT user@your-vps-ip"
echo "  2. Clone repo: git clone <repo-url> /opt/cortexia"
echo "  3. Generate certs: bash /opt/cortexia/scripts/generate-certs.sh YOUR_VPS_IP"
echo "  4. Configure: cp /opt/cortexia/.env.example /opt/cortexia/.env && vim /opt/cortexia/.env"
echo "  5. Launch: cd /opt/cortexia && docker compose up -d"
echo "  6. Set up backups: crontab -e  # add: 0 2 * * * /opt/cortexia/deploy/backup.sh"
