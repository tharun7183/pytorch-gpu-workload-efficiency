
import argparse, os, json, csv
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

from .metrics import synchronize_if_cuda, latency_stats
from .optimizations import apply_channels_last, maybe_compile, autocast_ctx
from .profilers import maybe_profile

def get_model(name: str, num_classes=1000):
    name = name.lower()
    if name == "resnet50":
        return models.resnet50(weights=None, num_classes=num_classes)
    if name == "efficientnet_b0":
        return models.efficientnet_b0(weights=None, num_classes=num_classes)
    raise ValueError(f"Unsupported model: {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=10)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--compile", action="store_true")

    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile-dir", default="outputs/profiler_traces")

    ap.add_argument("--out-csv", default="outputs/results.csv")
    ap.add_argument("--out-json", default="outputs/train_result.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model = get_model(args.model).train().to(device)
    if args.channels_last:
        model = apply_channels_last(model)
    model = maybe_compile(model, args.compile)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    x = torch.randn(args.batch_size, 3, 224, 224, device=device)
    y = torch.randint(0, 1000, (args.batch_size,), device=device)
    if args.channels_last:
        x = x.to(memory_format=torch.channels_last)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device == "cuda"))

    # Warmup
    for _ in range(args.warmup):
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(args.amp):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    synchronize_if_cuda()

    prof = maybe_profile(args.profile, args.profile_dir)
    it_ms = []
    total_images = 0

    if prof:
        prof.__enter__()

    for _ in range(args.steps):
        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(args.amp):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if device == "cuda":
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)
        else:
            import time
            t0 = time.perf_counter()
            optimizer.step()
            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000.0

        it_ms.append(ms)
        total_images += args.batch_size

        if prof:
            prof.step()

    if prof:
        prof.__exit__(None, None, None)

    stats = latency_stats(it_ms)
    total_time_s = sum(it_ms) / 1000.0
    throughput = total_images / total_time_s if total_time_s > 0 else None

    result = {
        "task": "train_micro",
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
