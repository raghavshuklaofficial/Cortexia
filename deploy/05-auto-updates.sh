#!/bin/bash
# Cortexia — Automatic Security Updates
# Configures unattended-upgrades for automatic security patches.
set -euo pipefail

echo "=== Automatic Security Updates ==="

apt-get install -y -qq unattended-upgrades apt-listchanges

cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
    "${distro_id}ESMApps:${distro_codename}-apps-security";
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF

cat > /etc/apt/apt.conf.d/20auto-upgrades << 'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
EOF

systemctl enable unattended-upgrades
systemctl start unattended-upgrades

echo "Automatic security updates enabled."
echo "  Security patches: daily"
echo "  Auto-clean: weekly"
echo "  Auto-reboot: disabled (manual reboot recommended after kernel updates)"
