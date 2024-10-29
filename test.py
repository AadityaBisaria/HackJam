from diffusers import DiffusionPipeline
import torch
from PIL import Image

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors = True, variant = "fp16")
pipe.to("cuda")

prompt = "Ancient Athenian Soldier"

images = pipe(prompt=prompt).images[0]
images.show()
images.save("ancient athenian soldier.png")  # Save the image to a file
print("Image saved.png'")