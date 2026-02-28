#!/bin/bash
# Cortexia — CrowdSec Community IPS
# Free intrusion prevention system with global botnet IP blocklists.
# Directly counters Mozi-class botnets by sharing attacker IPs globally.
set -euo pipefail

echo "=== CrowdSec Setup ==="

# Install CrowdSec
curl -s https://packagecloud.io/install/repositories/crowdsec/crowdsec/script.deb.sh | bash
apt-get install -y crowdsec crowdsec-firewall-bouncer-iptables

# Install detection collections
cscli collections install crowdsecurity/nginx
cscli collections install crowdsecurity/sshd
cscli collections install crowdsecurity/linux

# Restart to apply
systemctl enable crowdsec
systemctl restart crowdsec

echo ""
echo "CrowdSec enabled with community threat intelligence."
echo "Collections installed: nginx, sshd, linux"
echo ""
echo "Useful commands:"
echo "  cscli metrics           — View detection stats"
echo "  cscli decisions list    — View active bans"
echo "  cscli alerts list       — View security alerts"
echo "  cscli hub update        — Update threat intelligence"
