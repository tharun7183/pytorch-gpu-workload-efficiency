
import torch
import statistics

def synchronize_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def latency_stats(latencies_ms):
    latencies_ms = list(latencies_ms)
    latencies_ms.sort()

    def pct(p):
        if not latencies_ms:
            return None
        k = int(round((p/100) * (len(latencies_ms)-1)))
        return latencies_ms[k]

    return {
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "mean_ms": statistics.mean(latencies_ms) if latencies_ms else None
    }
