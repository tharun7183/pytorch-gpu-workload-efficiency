
import os, subprocess
import pandas as pd

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
MODEL = "resnet50"

OUT = "outputs/sweep_results.csv"
os.makedirs("outputs", exist_ok=True)

if os.path.exists(OUT):
    os.remove(OUT)

def run_one(bs, amp=False, channels_last=False, compile_=False, tag=""):
    cmd = [
        "python", "-m", "src.benchmark_infer",
        "--model", MODEL,
        "--batch-size", str(bs),
        "--steps", "120",
        "--warmup", "20",
        "--out-csv", OUT,
        "--out-json", f"outputs/sweep_{tag}_bs{bs}.json",
    ]
    if amp:
        cmd.append("--amp")
    if channels_last:
        cmd.append("--channels-last")
    if compile_:
        cmd.append("--compile")

    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# baseline sweep
for bs in BATCH_SIZES:
    run_one(bs, amp=False, channels_last=False, compile_=False, tag="baseline")

# optimized sweep
for bs in BATCH_SIZES:
    run_one(bs, amp=True, channels_last=True, compile_=True, tag="optimized")

df = pd.read_csv(OUT).sort_values(["amp", "channels_last", "compile", "batch_size"])
print(df)
print("\nSaved:", OUT)
