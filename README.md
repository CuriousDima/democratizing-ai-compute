# Democratizing AI Compute

## Overview

The repository contains a few experiments based on the materials from https://www.modular.com/democratizing-ai-compute.

## Experiment: Inference with MAX

### [MAX](https://docs.modular.com/get-started/) Quickstart

Serving an LLM via `max`:

```bash
max serve \
  --model Qwen/Qwen3-8B \
  --devices=gpu:1 \
  --device-memory-utilization 0.9
```

`--device-memory-utilization` tells MAX what fraction of the currently available GPU memory it is allowed to plan to consume.

`KV cache workspace = (free GPU memory × device_memory_utilization) - model weights`

for example for 32GB GPU memory and Qwen/Qwen3-8B:

```
Nominal VRAM                  32.00 GB
× device-memory-utilization   0.90
───────────────────────────────────
MAX memory budget            28.80 GB

Qwen3-8B BF16 weights       -16.38 GB
───────────────────────────────────
Remaining                    12.42 GB
```

That 12.42 GB is not all guaranteed KV cache: MAX also needs some runtime/workspace memory.

How much does Qwen3-8B's KV cache need? The model is:

```
36 layers
8 KV heads
128 dimensions/head
BF16 = 2 bytes
```

Therefore KV cache per token is:
```
K + V
= 2 × 36 layers × 8 KV heads × 128 dims × 2 bytes
= 147,456 bytes/token
= 144 KiB/token
```

As a result, we are getting:

| Total cached tokens | BF16 KV cache |
| ------------------: | ------------: |
|               8,192 | **1.125 GiB** |
|              16,384 |  **2.25 GiB** |
|              32,768 |  **4.50 GiB** |
|              40,960 | **5.625 GiB** |

The benchmark start command:

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

## Experiment: General Matrix Multiply with `nvmath-python`

CUDA platform is vast. Beyond language/compiler/profiler it includes libraries, runtime, and utilities.
As a practical example, lets look at [nvmath](https://developer.nvidia.com/cuda/cuda-x-libraries) from CUDA X Libraries.

Notebook: nvmath-example.ipynb
