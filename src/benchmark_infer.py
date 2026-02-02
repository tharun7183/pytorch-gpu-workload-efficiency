
import argparse, os, json, csv
import torch
import torchvision.models as models

from .metrics import synchronize_if_cuda, latency_stats
from .optimizations import apply_channels_last, maybe_compile, autocast_ctx
from .profilers import maybe_profile

def get_model(name: str):
    name = name.lower()
    if name == "resnet50":
        return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    if name == "efficientnet_b0":
        return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    raise ValueError(f"Unsupported model: {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=20)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--compile", action="store_true")

    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile-dir", default="outputs/profiler_traces")

    ap.add_argument("--out-csv", default="outputs/results.csv")
    ap.add_argument("--out-json", default="outputs/result.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(args.model).eval().to(device)
    if args.channels_last:
        model = apply_channels_last(model)
    model = maybe_compile(model, args.compile)

    x = torch.randn(args.batch_size, 3, 224, 224, device=device)
    if args.channels_last:
        x = x.to(memory_format=torch.channels_last)

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            with autocast_ctx(args.amp):
                _ = model(x)
        synchronize_if_cuda()

    prof = maybe_profile(args.profile, args.profile_dir)
    latencies = []
    total_images = 0

    with torch.no_grad():
        if prof:
            prof.__enter__()

        for _ in range(args.steps):
            if device == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                with autocast_ctx(args.amp):
                    _ = model(x)

                end.record()
                torch.cuda.synchronize()
                ms = start.elapsed_time(end)
            else:
                import time
                t0 = time.perf_counter()
                with autocast_ctx(args.amp):
                    _ = model(x)
                t1 = time.perf_counter()
                ms = (t1 - t0) * 1000.0

            latencies.append(ms)
            total_images += args.batch_size

            if prof:
                prof.step()

        if prof:
            prof.__exit__(None, None, None)

    stats = latency_stats(latencies)
    total_time_s = sum(latencies) / 1000.0
    throughput = total_images / total_time_s if total_time_s > 0 else None

    result = {
        "task": "inference",
        "model": args.model,
        "device": device,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "amp": args.amp,
        "channels_last": args.channels_last,
        "compile": args.compile,
        "throughput_img_s": throughput,
        **stats,
    }

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result.keys()))
        if write_header:
            w.writeheader()
        w.writerow(result)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
