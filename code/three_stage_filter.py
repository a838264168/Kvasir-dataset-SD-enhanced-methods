#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-Stage SD Image Filter for Binary Classification
As described in the paper: "Progressive Training with Filtered Synthetic Data for Medical Image Classification"

Stage 1: Quality + Semantic (50% keep)
  - Laplacian variance (sharpness) threshold: >100
  - CLIP similarity to prompt (typically >0.7 for high-quality matches)
  - Weighted combination: 60% quality, 40% semantics
  - Output: Top 50% (1,250 images from 2,500 candidates)

Stage 2: Domain Similarity (50% keep)
  - ResNet50 feature extraction (ImageNet pretrained)
  - Cosine similarity with real data centroid (computed from 500 randomly sampled real images)
  - Similarity threshold: typically >0.75
  - Output: Top 50% (625 images from 1,250)

Stage 3: Diversity (KMeans, final 200 images)
  - K-Means clustering with K=200
  - Select image closest to each cluster centroid
  - Output: 200 diverse, high-quality synthetic images
"""

import os
import shutil
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from torchvision import models, transforms
from tqdm import tqdm

# ===== Configuration =====
CATEGORY = "disease"  # or "normal"
REAL_DIR = f"../datasets_binary/baseline_real_only/train/{CATEGORY}"
SD_DIR = f"../datasets_sd21_filtered_v2/{CATEGORY}"  # Input: candidate images from generate_sd15_dataset.py
OUT_DIR = f"../datasets_sd21_filtered_v2/{CATEGORY}"  # Output: filtered images
NUM_FINAL = 200  # final selected images (as per paper)
SEED = 42
STAGE1_KEEP_RATIO = 0.5  # Keep top 50% in Stage 1
STAGE2_KEEP_RATIO = 0.5  # Keep top 50% in Stage 2
NUM_REAL_SAMPLES = 500  # Number of real images to sample for centroid computation

# ===== Load Models =====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print("Loading CLIP model (ViT-B/32)...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Loading ResNet50 feature extractor (ImageNet pretrained)...")
feature_extractor = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1]).eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ===== Helper Functions =====
def load_images(dir_path):
    """Load all image files from directory"""
    dir_p = Path(dir_path)
    if not dir_p.exists():
        print(f"Warning: {dir_path} does not exist!")
        return []
    files = [f for f in dir_p.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    return sorted(files)

def compute_clip_score(image_path, prompt):
    """Compute CLIP semantic similarity score"""
    try:
        img = Image.open(image_path).convert('RGB')
        inputs = clip_proc(text=[prompt], images=img, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            # Use cosine similarity between image and text embeddings
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            score = torch.nn.functional.cosine_similarity(image_embeds, text_embeds).item()
        return score
    except Exception as e:
        print(f"Error computing CLIP score for {image_path}: {e}")
        return 0.0

def compute_sharpness(image_path):
    """Compute Laplacian variance (sharpness) - threshold >100 for sharp images"""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        lap = cv2.Laplacian(img, cv2.CV_64F).var()
        return float(lap)
    except Exception as e:
        print(f"Error computing sharpness for {image_path}: {e}")
        return 0.0

def extract_features(image_path):
    """Extract ResNet50 features (2048 dimensions)"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = feature_extractor(img_tensor).squeeze().cpu().numpy().flatten()
        # L2 normalization as per paper
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        return feat
    except Exception as e:
        print(f"Error extracting features for {image_path}: {e}")
        return np.zeros(2048)

# ===== Stage 1: Quality + Semantic =====
def stage1_quality_semantic(sd_images, prompt):
    """Stage 1: Filter by quality (sharpness) and semantic relevance (CLIP)"""
    print(f"\n{'='*60}")
    print(f"Stage 1: Quality + Semantic Filtering")
    print(f"{'='*60}")
    print(f"Input: {len(sd_images)} candidate images")
    
    scores = []
    for img_path in tqdm(sd_images, desc="Computing scores"):
        clip_score = compute_clip_score(img_path, prompt)
        sharp_score = compute_sharpness(img_path)
        scores.append({
            'path': img_path,
            'clip': clip_score,
            'sharp': sharp_score,
        })
    
    # Normalize sharpness scores
    sharp_vals = [s['sharp'] for s in scores]
    max_sharp = max(sharp_vals) if max(sharp_vals) > 0 else 1.0
    for s in scores:
        s['sharp_norm'] = s['sharp'] / max_sharp
    
    # Combined score: 60% quality (sharpness), 40% semantics (CLIP)
    # As per paper: "weight ratio: 0.6 for quality, 0.4 for semantics"
    for s in scores:
        s['combined'] = 0.6 * s['sharp_norm'] + 0.4 * s['clip']
    
    # Sort by combined score
    scores.sort(key=lambda x: x['combined'], reverse=True)
    
    # Keep top 50% (1,250 from 2,500 as per paper)
    num_keep = max(1, int(len(scores) * STAGE1_KEEP_RATIO))
    selected = [s['path'] for s in scores[:num_keep]]
    
    print(f"Output: {len(selected)} images (top {int(STAGE1_KEEP_RATIO*100)}%)")
    print(f"Sharpness range: {min(sharp_vals):.2f} - {max(sharp_vals):.2f}")
    print(f"CLIP similarity range: {min([s['clip'] for s in scores]):.4f} - {max([s['clip'] for s in scores]):.4f}")
    return selected, scores[:num_keep]

# ===== Stage 2: Domain Similarity =====
def stage2_domain_similarity(selected_images, real_images, prompt):
    """Stage 2: Filter by domain similarity (cosine similarity with real data centroid)"""
    print(f"\n{'='*60}")
    print(f"Stage 2: Domain Similarity Filtering")
    print(f"{'='*60}")
    
    # Sample real images if too many (paper uses 500 randomly sampled)
    if len(real_images) > NUM_REAL_SAMPLES:
        import random
        random.seed(SEED)
        real_images = random.sample(real_images, NUM_REAL_SAMPLES)
        print(f"Sampling {len(real_images)} real images for centroid computation")
    
    # Extract features from real images
    print(f"Extracting features from {len(real_images)} real images...")
    real_feats = []
    for img_path in tqdm(real_images, desc="Processing real images"):
        feat = extract_features(img_path)
        if feat is not None and feat.sum() != 0:
            real_feats.append(feat)
    
    if len(real_feats) == 0:
        print("Warning: No valid real image features! Skipping stage 2.")
        return selected_images
    
    # Compute real data centroid (mean feature vector)
    real_centroid = np.mean(real_feats, axis=0)
    # L2 normalize centroid
    real_centroid = real_centroid / (np.linalg.norm(real_centroid) + 1e-8)
    print(f"Real data centroid shape: {real_centroid.shape}")
    
    # Extract features from SD images
    print(f"Extracting features from {len(selected_images)} SD images...")
    sd_feats = []
    valid_paths = []
    for img_path in tqdm(selected_images, desc="Processing SD images"):
        feat = extract_features(img_path)
        if feat is not None and feat.sum() != 0:
            sd_feats.append(feat)
            valid_paths.append(img_path)
    
    if len(sd_feats) == 0:
        print("Warning: No valid SD image features! Skipping stage 2.")
        return selected_images
    
    # Compute cosine similarity with real centroid
    sd_feats_array = np.array(sd_feats)
    similarities = cosine_similarity(sd_feats_array, [real_centroid]).flatten()
    
    # Pair with paths and sort
    paired = list(zip(similarities, valid_paths))
    paired.sort(reverse=True)
    
    # Keep top 50% (625 from 1,250 as per paper)
    num_keep = max(1, int(len(paired) * STAGE2_KEEP_RATIO))
    selected = [img_path for sim, img_path in paired[:num_keep]]
    
    print(f"Output: {len(selected)} images (top {int(STAGE2_KEEP_RATIO*100)}% by domain similarity)")
    print(f"Similarity range: {similarities.min():.4f} - {similarities.max():.4f}")
    print(f"Typical threshold: >0.75 for high domain similarity")
    return selected

# ===== Stage 3: Diversity (KMeans) =====
def stage3_diversity(selected_images, num_final):
    """Stage 3: Select diverse samples using KMeans clustering"""
    print(f"\n{'='*60}")
    print(f"Stage 3: Diversity Selection (KMeans)")
    print(f"{'='*60}")
    print(f"Input: {len(selected_images)} images")
    
    # Extract features
    feats = []
    valid_paths = []
    for img_path in tqdm(selected_images, desc="Extracting features"):
        feat = extract_features(img_path)
        if feat is not None and feat.sum() != 0:
            feats.append(feat)
            valid_paths.append(img_path)
    
    if len(feats) == 0:
        print("Warning: No valid features! Returning all images.")
        return selected_images[:num_final]
    
    feats = np.array(feats)
    print(f"Feature matrix shape: {feats.shape}")
    
    # KMeans clustering (K=200 as per paper)
    num_clusters = min(num_final, len(feats))
    print(f"Clustering into {num_clusters} clusters (K-Means with k-means++ initialization)...")
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=SEED,
        n_init=10,
        init='k-means++',  # As per paper: "k-means++ initialization"
        max_iter=300,      # As per paper: "up to 300 iterations"
        tol=1e-4,          # As per paper: "tolerance: 1e-4"
    )
    kmeans.fit(feats)
    
    # Select representative from each cluster (closest to centroid)
    representatives = []
    for i in range(num_clusters):
        cluster_mask = kmeans.labels_ == i
        cluster_feats = feats[cluster_mask]
        cluster_paths = [valid_paths[j] for j in range(len(valid_paths)) if cluster_mask[j]]
        
        if len(cluster_feats) == 0:
            continue
        
        # Find closest to cluster center (Euclidean distance in 2048-dimensional feature space)
        center = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(cluster_feats - center, axis=1)
        closest_idx = np.argmin(distances)
        representatives.append(cluster_paths[closest_idx])
    
    print(f"Output: {len(representatives)} diverse images")
    return representatives[:num_final]

# ===== Main =====
def main():
    print(f"\n{'='*60}")
    print(f"Three-Stage SD Image Filter")
    print(f"Category: {CATEGORY}")
    print(f"Final target: {NUM_FINAL} images")
    print(f"{'='*60}\n")
    
    # Load images
    sd_images = load_images(SD_DIR)
    real_images = load_images(REAL_DIR)
    
    if len(sd_images) == 0:
        print(f"Error: No SD candidate images found in {SD_DIR}")
        print(f"Please run generate_sd15_dataset.py first to generate candidate images.")
        return
    
    if len(real_images) == 0:
        print(f"Error: No real images found in {REAL_DIR}")
        return
    
    print(f"Found {len(sd_images)} SD candidate images")
    print(f"Found {len(real_images)} real images")
    
    # Prompts for CLIP (as per paper: domain-specific prompt engineering)
    prompts = {
        "normal": "clinical endoscopic photograph, high-resolution medical imaging, realistic mucosal textures, professional medical photography, soft lighting, natural colors, detailed textures, clinical quality, medical imaging standard",
        "disease": "clinical endoscopic photograph, high-resolution medical imaging, realistic mucosal textures, professional medical photography, soft lighting, natural colors, detailed textures, clinical quality, medical imaging standard, visible lesions or inflammation",
    }
    prompt = prompts.get(CATEGORY, prompts["disease"])
    
    # Stage 1: Quality + Semantic
    stage1_selected, stage1_scores = stage1_quality_semantic(sd_images, prompt)
    
    # Stage 2: Domain Similarity
    stage2_selected = stage2_domain_similarity(stage1_selected, real_images, prompt)
    
    # Stage 3: Diversity
    final_selected = stage3_diversity(stage2_selected, NUM_FINAL)
    
    # Save selected images
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Saving {len(final_selected)} selected images to {OUT_DIR}")
    print(f"{'='*60}\n")
    
    # Clear existing filtered images in output directory (optional)
    # Uncomment the next 3 lines if you want to clear the directory first
    # for existing_file in out_dir.glob(f"{CATEGORY}_filtered_*.png"):
    #     existing_file.unlink()
    
    for i, img_path in enumerate(tqdm(final_selected, desc="Copying images")):
        shutil.copy2(str(img_path), str(out_dir / f"{CATEGORY}_filtered_{i:05d}{img_path.suffix}"))
    
    print(f"\n✅ Done! Selected images saved to: {OUT_DIR}")
    print(f"   Final count: {len(final_selected)} images")
    print(f"\nNext step: Use these filtered images for progressive training with train_ablation_combo_full.py")

if __name__ == "__main__":
    main()

