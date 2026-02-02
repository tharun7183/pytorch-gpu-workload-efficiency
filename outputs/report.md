# GPU Workload Efficiency Benchmark Report

This report summarizes inference and micro-training benchmarks with optimization toggles.

| task        | model    | device   |   batch_size | amp   | channels_last   | compile   |   throughput_img_s |   p50_ms |   p95_ms |   mean_ms |
|:------------|:---------|:---------|-------------:|:------|:----------------|:----------|-------------------:|---------:|---------:|----------:|
| inference   | resnet50 | cuda     |           32 | True  | True            | True      |           1506.46  |  21.2787 |  21.7047 |   21.2418 |
| inference   | resnet50 | cuda     |           32 | True  | True            | False     |            971.872 |  32.8986 |  33.3029 |   32.9261 |
| inference   | resnet50 | cuda     |           32 | True  | False           | False     |            814.464 |  39.2253 |  39.9361 |   39.2897 |
| inference   | resnet50 | cuda     |           32 | False | False           | False     |            383.788 |  83.3743 |  84.3067 |   83.3794 |
| train_micro | resnet50 | cuda     |           32 | True  | True            | False     |            346.059 |  92.4172 |  93.3108 |   92.4698 |
| train_micro | resnet50 | cuda     |           32 | False | False           | False     |            108.69  | 294.484  | 297.83   |  294.414  |

## Notes
- Higher throughput_img_s is better. Lower latency (p50/p95) is better.
- Profiling traces are available under outputs/profiler_traces/.
