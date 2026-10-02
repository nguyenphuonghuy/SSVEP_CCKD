#!/bin/bash
# Nguon file nay de co 2 ham pi_ssh() va pi_scp() dung ket noi Raspberry Pi 4
# qua Tailscale tu HPC node nay:
#
#     source pi4_latency/connect.sh
#     pi_ssh 'hostname && uname -a'
#     pi_scp local_file.py "$PI4_USER@$PI4_IP:~/ssvep_latency_pkg/"
#
# Vi sao khong ping/ssh truc tiep duoc: tailscaled cua tai khoan nay chay o
# che do userspace-networking (khong co interface kernel tailscale0), nen phai
# di qua lenh `tailscale nc` (tsnet built-in, khong can mo cong SOCKS/HTTP hay
# restart tailscaled) lam ProxyCommand cho ssh/scp. Xem README.md muc "Vi sao
# khong SSH truc tiep duoc" de biet chi tiet qua trinh chan doan.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRED_FILE="$SCRIPT_DIR/.credentials.local"

if [ ! -f "$CRED_FILE" ]; then
    echo "Khong tim thay $CRED_FILE — xem README.md muc 'Chuan bi credentials'." >&2
    return 1 2>/dev/null || exit 1
fi
source "$CRED_FILE"

if ! command -v sshpass >/dev/null 2>&1; then
    echo "Thieu lenh sshpass (can de dang nhap bang mat khau qua script)." >&2
    return 1 2>/dev/null || exit 1
fi

pi_ssh() {
    sshpass -p "$PI4_PASS" ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new \
        -o ProxyCommand="$TS_BIN --socket=$TS_SOCK nc $PI4_IP 22" \
        "$PI4_USER@$PI4_IP" "$@"
}

pi_scp() {
    sshpass -p "$PI4_PASS" scp -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new \
        -o ProxyCommand="$TS_BIN --socket=$TS_SOCK nc $PI4_IP 22" \
        "$@"
}

echo "OK: pi_ssh() va pi_scp() da san sang (target: $PI4_USER@$PI4_IP)."
