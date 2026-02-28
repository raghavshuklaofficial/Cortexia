#!/bin/bash
# Cortexia — Fail2ban Setup
# Protects SSH and nginx from brute force and rate limit abuse.
set -euo pipefail

SSH_PORT="${1:-2222}"

echo "=== Fail2ban Setup ==="

apt-get install -y -qq fail2ban

cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3
banaction = ufw

[sshd]
enabled = true
port = $SSH_PORT
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 86400

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
logpath = /var/log/nginx/error.log
maxretry = 5

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
logpath = /var/log/nginx/error.log
maxretry = 10
findtime = 60
bantime = 3600

[nginx-botsearch]
enabled = true
filter = nginx-botsearch
logpath = /var/log/nginx/access.log
maxretry = 2
bantime = 86400
EOF

systemctl enable fail2ban
systemctl restart fail2ban

echo ""
echo "Fail2ban enabled. Status:"
fail2ban-client status
