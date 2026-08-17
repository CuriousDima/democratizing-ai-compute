# Democratizing AI Compute

## Overview

The repository contains a few experiments based on the materials from https://www.modular.com/democratizing-ai-compute.

## Experiment: Inference with MAX

### [MAX](https://docs.modular.com/get-started/) Quickstart

To serve an LLM with `max`:

```bash
max serve \
  --model Qwen/Qwen3-8B \
  --devices=gpu:1 \
  --device-memory-utilization 0.9
```

`--device-memory-utilization` tells MAX what fraction of the currently available GPU memory it is allowed to plan to consume.

`KV cache workspace = (free GPU memory × device_memory_utilization) - model weights`

For example, with 32 GB of GPU memory and Qwen/Qwen3-8B:

```
Nominal VRAM                  32.00 GB
× device-memory-utilization   0.90
───────────────────────────────────
MAX memory budget            28.80 GB

Qwen3-8B BF16 weights       -16.38 GB
───────────────────────────────────
Remaining                    12.42 GB
```

That 12.42 GB is not guaranteed to be entirely available for the KV cache: MAX also needs some runtime and workspace memory.

How much memory does Qwen3-8B's KV cache require per token? The model has:

```
36 layers
8 KV heads
128 dimensions/head
BF16 = 2 bytes
```

Therefore, the KV cache size per token is:

```
K + V
= 2 × 36 layers × 8 KV heads × 128 dims × 2 bytes
= 147,456 bytes/token
= 144 KiB/token
```

This yields:

| Total cached tokens | BF16 KV cache |
| ------------------: | ------------: |
|               8,192 | **1.125 GiB** |
|              16,384 |  **2.25 GiB** |
|              32,768 |  **4.50 GiB** |
|              40,960 | **5.625 GiB** |

To run the benchmark:

```bash
max benchmark \
  --model Qwen/Qwen3-8B \
  --backend modular \
  --endpoint /v1/chat/completions \
  --dataset-name sonnet \
  --num-prompts 500 \
  --sonnet-input-len 550 \
  --output-lengths 256 \
  --sonnet-prefix-len 200 \
  --max-concurrency 32 \
  --result-filename "quickstart-qwen3-8b-benchmark.json"
```

## Experiment: General Matrix Multiplication with `nvmath-python`

The CUDA platform is vast. In addition to languages, compilers, and profiling tools, it includes libraries, runtimes, and utilities.
As a practical example, let's look at [nvmath](https://developer.nvidia.com/cuda/cuda-x-libraries) from the CUDA-X Libraries.

Notebook: [nvmath-example.ipynb](nvmath-example.ipynb)
