# CAB: Accelerating Flow and Diffusion Sampling via Rectification and Corrected Adams-Bashforth

Official implementation of CAB-2 and CAB-3 for stable low-NFE sampling in diffusion and flow-matching models.

This repository contains modified implementations for:

- EDM  
  https://github.com/NVlabs/edm

- DiT  
  https://github.com/facebookresearch/DiT

- HuggingFace Diffusers  
  https://github.com/huggingface/diffusers

---

# Repository Structure

```text
CAB/
├── README.md
├── gaussian_diffusion.py
├── generate_cab.py
├── pipeline_qwenimage.py
├── qwen.py
├── sample.py
└── scheduling_cab.py
```

---

# Datasets and Evaluation

## CIFAR-10

Used for EDM experiments.

Dataset:  
https://www.cs.toronto.edu/~kriz/cifar.html

---

## ImageNet 64×64

Used for EDM ImageNet-64 experiments.

Dataset:  
https://image-net.org/download-images.php

---

## ImageNet 256×256

Used for DiT experiments.

Dataset:  
https://image-net.org/index.php

DiT preprocessing / setup:  
https://github.com/facebookresearch/DiT

---

## MS-COCO

Used for Qwen-Image text-to-image evaluation.

Dataset:  
https://cocodataset.org/#download

Captions:  
https://cocodataset.org/#captions-2015

---

## EvalCrafter

Used for text-video alignment, temporal consistency, and visual quality evaluation.

Repository:  
https://github.com/evalcrafter/EvalCrafter

Paper:  
https://arxiv.org/abs/2310.11440

---

# 1. EDM Integration

Base repository:

https://github.com/NVlabs/edm

---

## File Replacement

Instead of the original EDM file:

```text
generate.py
```

use:

```text
generate_cab.py
```

provided in this repository.

---

## CIFAR-10 Sampling Example

### CAB-2 Example

```bash
python generate_cab.py \
    --outdir=out_ab2 \
    --seeds=0-63 \
    --batch=64 \
    --order=2 \
    --theta=0.9 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
```

---

### CAB-3 Example

```bash
python generate_cab.py \
    --outdir=out_ab3 \
    --seeds=0-63 \
    --batch=64 \
    --order=3 \
    --theta=0.9 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
```

---

# 2. DiT Integration

Base repository:

https://github.com/facebookresearch/DiT

---

## File Replacement

Inside the original DiT repository:

### Replace

```text
diffusion/gaussian_diffusion.py
```

with given:

```text
gaussian_diffusion.py
```

from this repository.

---

### Replace

```text
sample.py
```

with given:

```text
sample.py
```

from this repository.

---

## DiT Sampling

### CAB-2

```bash
python sample.py \
    --image-size 256 \
    --seed 1 \
    --order 2 \
    --theta 0.9
```

---

### CAB-3

```bash
python sample.py \
    --image-size 256 \
    --seed 1 \
    --order 3 \
    --theta 0.9
```

---

# 3. Qwen-Image and HunyuanVideo / Diffusers Integration

Base repository:

https://github.com/huggingface/diffusers

---

## Scheduler Integration

Inside the Diffusers repository:

```text
src/diffusers/schedulers/
```

add:

```text
scheduling_cab.py
```

from this repository.

The scheduler can then be imported as:

```python
from diffusers import CABScheduler
```

---

## Qwen-Image Pipeline Replacement

Inside:

```text
src/diffusers/pipelines/qwenimage/
```

replace:

```text
pipeline_qwenimage.py
```

with the modified version provided in this repository:

```text
pipeline_qwenimage.py
```

---

## Example Qwen-Image Sampling Script

Example script:

```text
qwen.py
```

is provided in this repository.

---

## Example Usage in qwen.py

```python
import os
import sys
import torch

# Use local diffusers source
sys.path.insert(0, "path/to/diffusers/src")

from diffusers import QwenImagePipeline, CABScheduler

model_name = "Qwen/Qwen-Image"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

pipe = QwenImagePipeline.from_pretrained(
    model_name,
    torch_dtype=dtype,
)

pipe.scheduler = CABScheduler.from_config(
    pipe.scheduler.config,
    solver_order=2,
    theta=0.2,
    prediction_type="flow_prediction",
    algorithm_type="cab",
    use_flow_sigmas=True,
)

pipe = pipe.to(device)

prompt = "A beautiful music room."

image = pipe(
    prompt=prompt,
    width=1024,
    height=1024,
    num_inference_steps=10,
    true_cfg_scale=4.0,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]

os.makedirs("outputs", exist_ok=True)
image.save("outputs/cab_qwen_music_room.png")
```

---

