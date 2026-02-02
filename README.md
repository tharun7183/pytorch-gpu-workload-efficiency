
# PyTorch GPU Workload Efficiency Benchmark Suite

Reproducible benchmarking + profiling toolkit for PyTorch training and inference on GPU.

## What it measures
- **Throughput:** images/sec
- **Latency:** p50 / p95 (ms)

## Quickstart (Colab)
Run the notebook `notebooks/GWE_Benchmarks_Colab.ipynb`.

## Profiling
Profiler traces are written to `outputs/profiler_traces/` and can be viewed in TensorBoard.

## Batch sweep plots
Plots are saved in `assets/`.
