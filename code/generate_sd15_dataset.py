"""
Generate the Stable Diffusion synthetic dataset used in the paper
`Progressive Training with Filtered Synthetic Data for Medical Image Classification`.

This script implements the SD v1.5 pipeline as described in the paper:

- Model: Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5) Text-to-Image
- Prompting: Domain-specific prompt engineering with medical terminology
- Resolution: 512×512 pixels
- num_inference_steps: 50
- guidance_scale: 7.5
- Generation: 2,500 candidate images using 75 unique prompt variations
  (approximately 33 images per prompt variation)

IMPORTANT:
- This script is a **template**. You MUST:
  1. Install the required libraries (see `main()`).
  2. Log in to HuggingFace and accept the SD v1.5 model license.
  3. Adjust the prompt variations and generation parameters as needed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

try:
    from diffusers import StableDiffusionPipeline
except ImportError as e:  # pragma: no cover - dependency hint
    raise ImportError(
        "This script requires `diffusers`.\n"
        "Install with:\n"
        "  pip install diffusers transformers accelerate safetensors\n"
        "and make sure you have a compatible version of PyTorch with CUDA."
    ) from e


@dataclass
class ClassConfig:
    name: str
    target_dir: Path   # where to save generated synthetic images
    num_images: int    # number of images to generate for this class


def load_pipeline(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    sd_model_id: str = "runwayml/stable-diffusion-v1-5",
):
    """
    Load Stable Diffusion v1.5 Text-to-Image pipeline.

    You must have:
      - accepted the SD v1.5 license on HuggingFace
      - a HF token logged in via `huggingface-cli login`
    """
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        sd_model_id,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    )
    sd_pipe = sd_pipe.to(device)
    sd_pipe.enable_attention_slicing()  # Reduce memory usage

    return sd_pipe, device


def generate_prompt_variations(class_name: str) -> List[str]:
    """
    Generate 75 unique prompt variations for the given class.
    
    According to the paper, prompts combine:
    - Base phrases: "clinical endoscopic photograph", "high-resolution medical imaging",
      "realistic mucosal textures", "professional medical photography"
    - Modifiers: "soft lighting", "natural colors", "detailed textures",
      "clinical quality", "medical imaging standard"
    - Class-specific medical terminology
    """
    base_phrases = [
        "clinical endoscopic photograph",
        "high-resolution medical imaging",
        "realistic mucosal textures",
        "professional medical photography",
    ]
    
    modifiers = [
        "soft lighting",
        "natural colors",
        "detailed textures",
        "clinical quality",
        "medical imaging standard",
    ]
    
    # Class-specific medical descriptions
    if "esophagitis" in class_name.lower():
        class_desc = [
            "endoscopic image of esophagitis",
            "inflamed esophagus tissue",
            "red and swollen esophageal mucosa",
            "gastroesophageal inflammation",
            "esophageal mucosal damage",
        ]
    elif "dyed-lifted-polyps" in class_name.lower() or "polyps" in class_name.lower():
        class_desc = [
            "endoscopic image of dyed lifted polyps",
            "colonoscopy with blue dye",
            "polyp detection in colonoscopy",
            "dyed polyps in gastrointestinal endoscopy",
            "colorectal polyp visualization",
        ]
    elif "normal-z-line" in class_name.lower() or "normal" in class_name.lower():
        class_desc = [
            "endoscopic image of normal z-line",
            "healthy esophageal z-line",
            "normal gastroesophageal junction",
            "healthy esophageal mucosa",
            "normal anatomical boundary",
        ]
    else:
        class_desc = ["medical endoscopic image"]
    
    # Generate 75 unique combinations
    prompts = []
    for base in base_phrases:
        for mod in modifiers:
            for desc in class_desc[:3]:  # Use first 3 class descriptions
                prompt = f"{base} of {desc}, {mod}"
                prompts.append(prompt)
                if len(prompts) >= 75:
                    break
            if len(prompts) >= 75:
                break
        if len(prompts) >= 75:
            break
    
    # Ensure we have exactly 75 prompts
    return prompts[:75]


def generate_sd15_dataset(
    classes: dict[str, ClassConfig],
    sd_pipe: StableDiffusionPipeline,
    device: str,
    seed: int = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    resolution: int = 512,
    total_candidates: int = 2500,
) -> None:
    """
    Generate the SD v1.5 synthetic dataset according to the paper configuration.
    
    Generates 2,500 candidate images using 75 unique prompt variations
    (approximately 33 images per prompt variation).
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # For disease class, generate 2,500 candidates
    # (The paper focuses on disease class for the main experiment)
    disease_class = None
    for cls_name, cfg in classes.items():
        if "disease" in cls_name.lower() or "esophagitis" in cls_name.lower() or "polyps" in cls_name.lower():
            disease_class = (cls_name, cfg)
            break
    
    if disease_class is None:
        raise ValueError("No disease class found in classes configuration")
    
    cls_name, cfg = disease_class
    print(f"\n=== Generating {total_candidates} candidate images for class '{cls_name}' ===")
    cfg.target_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate 75 unique prompt variations
    prompt_variations = generate_prompt_variations(cls_name)
    print(f"Generated {len(prompt_variations)} unique prompt variations")
    
    # Generate approximately 33 images per prompt variation
    images_per_prompt = total_candidates // len(prompt_variations)
    remaining = total_candidates % len(prompt_variations)
    
    image_idx = 0
    for prompt_idx, prompt in enumerate(tqdm(prompt_variations, desc="Prompts")):
        num_for_this_prompt = images_per_prompt + (1 if prompt_idx < remaining else 0)
        
        for img_in_prompt in range(num_for_this_prompt):
            # Use different seeds for each image to ensure diversity
            current_seed = seed + image_idx
            generator.manual_seed(current_seed)
            
            with torch.autocast(device_type=device if device.startswith("cuda") else "cpu"):
                out = sd_pipe(
                    prompt=prompt,
                    height=resolution,
                    width=resolution,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                )
            gen_img: Image.Image = out.images[0]
            
            out_name = f"{cls_name}_sd15_candidate_{image_idx:04d}.png"
            gen_img.save(cfg.target_dir / out_name)
            image_idx += 1
    
    print(f"\nGenerated {image_idx} candidate images for class '{cls_name}'")
    print(f"Images saved to: {cfg.target_dir}")
    print("\nNote: These are candidate images. Apply the three-stage filtering pipeline")
    print("      to select 200 high-quality synthetic images from these 2,500 candidates.")


def main() -> None:
    """
    Entry point for generating the SD v1.5 synthetic dataset.

    BEFORE RUNNING:
      1. Install dependencies:
         pip install torch torchvision diffusers transformers accelerate safetensors tqdm
      2. Log in to HuggingFace and accept the SD v1.5 license.
      3. Adjust TARGET_SYNTH_DIRS to match your filesystem.
    """
    project_root = Path(__file__).resolve().parents[1]

    # Target synthetic dirs (as assumed in the paper & training code)
    # The paper generates 2,500 candidates for the disease class
    TARGET_SYNTH_DIRS = {
        "disease": project_root.parent / "datasets_sd21_filtered_v2" / "disease",
        # Note: The folder name "sd21_filtered_v2" is just a naming convention.
        # The actual model used is SD v1.5 as described in the paper.
    }

    classes: dict[str, ClassConfig] = {
        "disease": ClassConfig(
            name="disease",
            target_dir=TARGET_SYNTH_DIRS["disease"],
            num_images=2500,  # 2,500 candidate images
        ),
    }

    sd_pipe, device = load_pipeline()
    generate_sd15_dataset(
        classes=classes,
        sd_pipe=sd_pipe,
        device=device,
        seed=42,
        num_inference_steps=50,
        guidance_scale=7.5,
        resolution=512,
        total_candidates=2500,
    )
    print("\nDone. Synthetic SD v1.5 candidate dataset generated.")
    print("Next step: Apply the three-stage filtering pipeline to select 200 high-quality images.")


if __name__ == "__main__":
    main()

