import time, sys
t0=time.time()
print(f"[{time.time()-t0:.2f}s] starting, importing torch...", flush=True)
import torch
print(f"[{time.time()-t0:.2f}s] torch imported, checking cuda.is_available()...", flush=True)
avail = torch.cuda.is_available()
print(f"[{time.time()-t0:.2f}s] cuda.is_available()={avail}", flush=True)
if avail:
    print(f"[{time.time()-t0:.2f}s] device name: {torch.cuda.get_device_name(0)}", flush=True)
    x = torch.randn(2000,2000, device='cuda')
    print(f"[{time.time()-t0:.2f}s] allocated tensor on cuda", flush=True)
    y = (x @ x).sum().item()
    print(f"[{time.time()-t0:.2f}s] matmul done, sum={y:.2f}", flush=True)
print(f"[{time.time()-t0:.2f}s] DONE OK", flush=True)
