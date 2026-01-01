"""
Example: Text-to-Image Generation using SanaPipeline
----------------------------------------------------
This script demonstrates how to run a Text-to-Image inference pipeline using
the Sana model. Modify the paths below to point to your configuration and
checkpoint directories. The output will be saved as a .png image.

Author: [Furkan Karakaya]
Repository: https://github.com/F-Karakaya/Sana-Text-to-Image
"""

import torch
from torchvision.utils import save_image
from app.sana_pipeline import SanaPipeline


# -----------------------------------------------------------
# Setup: device & seed
# -----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)


# -----------------------------------------------------------
# Load Sana configuration & model checkpoint
# -----------------------------------------------------------
config_path = "PATH/TO/Sana_600M_app.yaml"  # e.g., ./configs/sana_app_config/Sana_600M_app.yaml
checkpoint_path = "PATH/TO/Sana_600M_512px_MultiLing.pth"  # e.g., ./checkpoints/Sana_600M_512px_MultiLing.pth

pipeline = SanaPipeline(config_path)
pipeline.from_pretrained(checkpoint_path)


# -----------------------------------------------------------
# Text prompt input
# -----------------------------------------------------------
prompt = 'A two-headed cyberpunk cat with a bright neon sign that reads ‘Sana’'


# -----------------------------------------------------------
# Image generation
# -----------------------------------------------------------
image = pipeline(
    prompt=prompt,
    height=512,
    width=512,
    guidance_scale=5.0,
    pag_guidance_scale=2.0,
    num_inference_steps=18,
    generator=generator,
)


# -----------------------------------------------------------
# Save result
# -----------------------------------------------------------
output_path = "sana_512px_output.png"
save_image(image, output_path, nrow=1, normalize=True, value_range=(-1, 1))

print(f"Image generation completed successfully. Saved to: {output_path}")
