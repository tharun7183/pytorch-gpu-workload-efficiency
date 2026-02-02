
import os
import pandas as pd
import matplotlib.pyplot as plt

IN_CSV = "outputs/sweep_results.csv"
os.makedirs("assets", exist_ok=True)

df = pd.read_csv(IN_CSV)

# Throughput plot
plt.figure()
for (amp, ch, comp), g in df.groupby(["amp", "channels_last", "compile"]):
    g = g.sort_values("batch_size")
    label = f"amp={amp}, cl={ch}, compile={comp}"
    plt.plot(g["batch_size"], g["throughput_img_s"], marker="o", label=label)

plt.xscale("log", base=2)
plt.xlabel("Batch size")
plt.ylabel("Throughput (img/s)")
plt.title("ResNet-50 Inference Throughput vs Batch Size")
plt.grid(True)
plt.legend()
plt.savefig("assets/batch_sweep_throughput.png", dpi=200, bbox_inches="tight")
plt.close()

# p50 latency plot
plt.figure()
for (amp, ch, comp), g in df.groupby(["amp", "channels_last", "compile"]):
    g = g.sort_values("batch_size")
    label = f"amp={amp}, cl={ch}, compile={comp}"
    plt.plot(g["batch_size"], g["p50_ms"], marker="o", label=label)

plt.xscale("log", base=2)
plt.xlabel("Batch size")
plt.ylabel("p50 latency (ms)")
plt.title("ResNet-50 Inference p50 Latency vs Batch Size")
plt.grid(True)
plt.legend()
plt.savefig("assets/batch_sweep_latency.png", dpi=200, bbox_inches="tight")
plt.close()

print("Saved plots:")
print(" - assets/batch_sweep_throughput.png")
print(" - assets/batch_sweep_latency.png")
