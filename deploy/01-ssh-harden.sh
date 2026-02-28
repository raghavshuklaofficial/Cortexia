#!/bin/bash
# Cortexia — SSH Hardening Script
# Run as root on a fresh VPS. Ensure your SSH key is installed BEFORE running.
set -euo pipefail

SSH_PORT="${1:-2222}"

echo "=== SSH Hardening ==="
echo "Changing SSH port to: $SSH_PORT"

# Backup original config
cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak

# Disable root login
sed -i 's/^#*PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config

# Disable password authentication (key-only)
sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

# Change SSH port
sed -i "s/^#*Port.*/Port $SSH_PORT/" /etc/ssh/sshd_config

# Disable empty passwords
sed -i 's/^#*PermitEmptyPasswords.*/PermitEmptyPasswords no/' /etc/ssh/sshd_config

# Limit authentication attempts
grep -q "^MaxAuthTries" /etc/ssh/sshd_config && \
    sed -i 's/^MaxAuthTries.*/MaxAuthTries 3/' /etc/ssh/sshd_config || \
    echo "MaxAuthTries 3" >> /etc/ssh/sshd_config

# Disable X11 forwarding
sed -i 's/^#*X11Forwarding.*/X11Forwarding no/' /etc/ssh/sshd_config

# Idle timeout (5 minutes)
grep -q "^ClientAliveInterval" /etc/ssh/sshd_config && \
    sed -i 's/^ClientAliveInterval.*/ClientAliveInterval 300/' /etc/ssh/sshd_config || \
    echo "ClientAliveInterval 300" >> /etc/ssh/sshd_config

grep -q "^ClientAliveCountMax" /etc/ssh/sshd_config && \
    sed -i 's/^ClientAliveCountMax.*/ClientAliveCountMax 2/' /etc/ssh/sshd_config || \
    echo "ClientAliveCountMax 2" >> /etc/ssh/sshd_config

systemctl restart sshd

echo "SSH hardened successfully."
echo "  Port: $SSH_PORT"
echo "  Root login: DISABLED"
echo "  Password auth: DISABLED (key-only)"
echo "  Max auth tries: 3"
echo ""
echo "WARNING: Ensure your SSH key is installed before logging out!"
echo "Test in a NEW terminal: ssh -p $SSH_PORT user@your-vps-ip"
