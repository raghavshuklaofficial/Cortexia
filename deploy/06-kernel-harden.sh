#!/bin/bash
# Cortexia — Kernel & Service Hardening
# Disables unnecessary services and hardens kernel network parameters.
set -euo pipefail

echo "=== Kernel & Service Hardening ==="

# Disable common attack vector services
for svc in telnet.socket telnetd xinetd rpcbind avahi-daemon cups bluetooth; do
    if systemctl is-active --quiet "$svc" 2>/dev/null; then
        systemctl disable "$svc" && systemctl stop "$svc"
        echo "  Disabled: $svc"
    fi
done

# Kernel networking hardening
cat >> /etc/sysctl.d/99-cortexia-hardening.conf << 'EOF'
# === Cortexia Security Hardening ===

# Enable SYN cookies (anti-SYN flood)
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2

# Disable ICMP redirects (prevent MITM routing attacks)
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0

# Enable reverse path filtering (anti-spoofing)
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1

# Ignore ICMP broadcast requests
net.ipv4.icmp_echo_ignore_broadcasts = 1

# Log martian packets (packets with impossible source addresses)
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1

# Disable source routing
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv6.conf.default.accept_source_route = 0

# Ignore bogus ICMP error responses
net.ipv4.icmp_ignore_bogus_error_responses = 1
EOF

sysctl --system > /dev/null 2>&1

echo ""
echo "Kernel hardening applied."
echo "  SYN cookies: enabled"
echo "  ICMP redirects: disabled"
echo "  Reverse path filtering: enabled"
echo "  Martian logging: enabled"
echo "  Source routing: disabled"
