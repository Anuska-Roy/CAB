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

# CAB
pipe.scheduler = CABScheduler.from_config(
    pipe.scheduler.config,
    solver_order=2,
    theta=0.2,
    prediction_type="flow_prediction",
    algorithm_type="cab",
    use_flow_sigmas=True,
)

pipe = pipe.to(device)

# Prompt
prompt = "A beautiful music room."
negative_prompt = ""

# Image resolution
width, height = 1024, 1024

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=10,
    true_cfg_scale=4.0,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Save output
output_path = os.path.join(output_dir, "cab_qwen_music_room.png")
image.save(output_path)

print(f"Image saved to: {output_path}")