import os
import torch
from diffusers import StableDiffusionXLPipeline

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Path where you want to save/load the model
local_model_path = "./my_sdxl_model"
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# Check if model exists locally
if not os.path.exists(local_model_path):
    print(f"Model not found locally. Downloading from Hugging Face: {model_id}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # use fp16 to save VRAM
    )
    # Save the model locally for future use
    pipe.save_pretrained(local_model_path)
    print(f"Model saved locally at {local_model_path}")
else:
    print(f"Model found locally at {local_model_path}. Loading from disk.")

# Load the model from local folder
pipe_local = StableDiffusionXLPipeline.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,
).to(device)

# Optional: enable memory-efficient attention
pipe_local.enable_xformers_memory_efficient_attention()

# Generate an image

while ip:=input("Enter prompt :") != 'q':
    image = pipe_local(str(ip)).images[0]
    image.save("output.png")
    print("Image saved as output.png")