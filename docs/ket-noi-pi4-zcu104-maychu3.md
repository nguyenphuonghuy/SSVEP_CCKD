# Kết nối maychu3 (`gv_bvnam`) với Raspberry Pi 4 và ZCU104

Tài liệu này hướng dẫn cấu hình SSH bằng khoá từ tài khoản `gv_bvnam` trên HPC adminsm
tới Pi 4 và board ZCU104. Chạy mọi lệnh trên maychu3, theo đúng thứ tự.
Các bước chưa được chạy thử trên tài khoản này.

## Đường đi

| Thiết bị | User | Địa chỉ | Đường đi |
|---|---|---|---|
| Pi 4 | `khoinguyen` | 100.125.255.91 | HPC → Pi qua `tailscale nc` |
| `ubuntu-huy` | `phuonghuy` | 100.101.245.23 | HPC → máy host qua `tailscale nc` |
| ZCU104 | `xilinx` | 192.168.2.99 | HPC → `ubuntu-huy` → board (`ProxyJump`) |

Board không nằm trên Tailscale. Board chỉ nối Ethernet trực tiếp với `ubuntu-huy`, nên phải đi qua máy này.

## Bước 1: Kiểm Tailscale

```bash
whoami    # kỳ vọng: gv_bvnam
ps -u $USER -f | grep "[t]ailscaled --tun=userspace"
~/tailscale/tailscale --socket=$HOME/tailscale/tailscaled.sock status | grep -E "pi4|ubuntu-huy"
```

- Nếu daemon đang chạy thì không khởi động lại, vì việc này làm rớt các phiên SSH khác.
- Nếu daemon chưa chạy thì dùng `~/tailscale/start-tailscaled.sh`.
- Nếu `status` không thấy `pi4` hoặc `ubuntu-huy` thì dừng lại và báo chủ mạng Tailscale. Không tự đổi cấu hình Tailscale.

## Bước 2: Tạo khoá riêng

Bỏ qua bước này nếu `~/.ssh/id_ed25519_nhung` đã tồn tại.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_nhung -C "gv_bvnam-nhung"
```

## Bước 3: Thêm vào `~/.ssh/config`

Xem file `~/.ssh/config` trước. Nếu đã có `Host pi4`, `Host ubuntu-huy` hoặc `Host zcu104` thì sửa khối đó, không thêm trùng.

```
Host pi4
  HostName 100.125.255.91
  User khoinguyen
  IdentityFile ~/.ssh/id_ed25519_nhung
  IdentitiesOnly yes
  ProxyCommand $HOME/tailscale/tailscale --socket=$HOME/tailscale/tailscaled.sock nc %h %p

Host ubuntu-huy
  HostName 100.101.245.23
  User phuonghuy
  IdentityFile ~/.ssh/id_ed25519_nhung
  IdentitiesOnly yes
  ProxyCommand $HOME/tailscale/tailscale --socket=$HOME/tailscale/tailscaled.sock nc %h %p

Host zcu104
  HostName 192.168.2.99
  User xilinx
  IdentityFile ~/.ssh/id_ed25519_nhung
  IdentitiesOnly yes
  ProxyJump ubuntu-huy
```

Sau đó đặt quyền cho file:

```bash
chmod 600 ~/.ssh/config
```

## Bước 4: Cài khoá lên thiết bị

Tác giả tự chạy các lệnh này trong terminal và tự gõ mật khẩu. Không chép mật khẩu vào chat, log, skill hay git.
Giữ đúng thứ tự, vì lệnh cuối đi qua `ubuntu-huy`.

```bash
ssh-copy-id -i ~/.ssh/id_ed25519_nhung.pub -o StrictHostKeyChecking=accept-new pi4
ssh-copy-id -i ~/.ssh/id_ed25519_nhung.pub -o StrictHostKeyChecking=accept-new ubuntu-huy
ssh-copy-id -i ~/.ssh/id_ed25519_nhung.pub -o StrictHostKeyChecking=accept-new zcu104
```

User Pi 4 `khoinguyen` lấy từ `Pi4_README.md`, chưa xác nhận lại. Nếu lệnh đầu báo sai user thì hỏi tác giả, rồi sửa khối `Host pi4`.

## Bước 5: Kiểm tra

```bash
ssh -o BatchMode=yes pi4 'hostname; nproc'                # kỳ vọng: huy / 4
ssh -o BatchMode=yes zcu104 'hostname; uname -m; nproc'   # kỳ vọng: pynq / aarch64 / 4
```

Luôn thêm `-o BatchMode=yes` khi dùng về sau. Nếu lệnh báo lỗi vì cần mật khẩu thì khoá đã hỏng: dừng lại và báo tác giả, không thử mật khẩu.

## Xử lý lỗi

| Triệu chứng | Xử lý |
|---|---|
| `ssh zcu104` lỗi | Thử `ssh ubuntu-huy hostname`. Nếu lỗi thì kiểm Tailscale. Nếu được thì chạy `ssh ubuntu-huy 'ping -c1 -W1 192.168.2.99'` để kiểm board. |
| Board báo `Permission denied (publickey...)` dù đã có khoá | Home trên PYNQ để quyền quá rộng. Chạy `chmod go-w ~; chmod 700 ~/.ssh` trên board. |
| `Host key verification failed` | Thêm `-o StrictHostKeyChecking=accept-new` một lần. |

## Lưu ý khi dùng

- `ubuntu-huy` phải bật và board phải nối dây thì mới vào được ZCU104.
- Trước khi đo, `uptime` trên thiết bị phải cho load gần 0. Với ZCU104, `ubuntu-huy` cũng không được chạy việc nặng (ví dụ Vivado synthesis).
- Không chạy 2 sweep song song. Các tài khoản HPC dùng chung thiết bị, nên hỏi trước khi đo.
- ZCU104 không có Internet: không `pip install` trực tiếp trên board, cài offline từ `wheels/`.
- Đồng hồ ZCU104 sai, nên không dùng timestamp trên board để ghi ngày đo.
- Quy trình đo chi tiết: skill `do-latency-nhung`.
