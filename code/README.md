## Project Overview: LLM-Enhanced Medical Image Classification

This project investigates how to leverage **filtered synthetic endoscopic images + progressive training** to improve the performance of medical image binary classification (Kvasir dataset: Normal vs Disease), and provides complete Python code to enable others to reproduce experiments and generate synthetic images from scratch without relying on pre-generated figure files.

### 1. Project Structure (Code Reproduction Related)

- **Experimental Code**
  - `code/train_ablation_combo_full.py`:
    - Includes 4 models (SimpleCNN, ResNet18, MobileNetV2, EfficientNetB0);
    - 32 configurations (Real / SD / SMOTE / Optuna / Progressive Training);
    - Trains from scratch and produces all experimental results and ablation analysis from the paper.
  - `code/generate_sd15_dataset.py`:
    - Generates synthetic endoscopic image datasets using Stable Diffusion v1.5;
    - Implements the prompt engineering strategy described in the paper (75 prompt variations, generating 2,500 candidate images);
    - Parameter configuration matches the paper (512×512 resolution, guidance_scale=7.5, num_inference_steps=50).
  - `code/three_stage_filter.py`:
    - Implements the three-stage filtering pipeline described in the paper, selecting 200 high-quality images from 2,500 candidates;
    - Stage 1: Quality + Semantic filtering (Laplacian variance + CLIP similarity, keep top 50%);
    - Stage 2: Domain similarity filtering (ResNet50 features with cosine similarity to real data centroid, keep top 50%);
    - Stage 3: Diversity selection (K-Means clustering, K=200, select representative image from each cluster).

### 2. Data and Path Conventions

The project assumes the following data directory structure (relative to `MCP_MAKE_9.11`):

- Real Kvasir binary classification data (training set)
  - `../datasets_binary/baseline_real_only/train/normal`
  - `../datasets_binary/baseline_real_only/train/disease`
- Stable Diffusion generated synthetic data (three-stage filtered version)
  - `../datasets_sd21_filtered_v2/disease`
  - `../datasets/medical_generated_sd21/normal-z-line` (normal-like synthetic images)

To reproduce the experiments, please adjust the above paths according to your data locations, or modify the paths in the code to match your actual setup.

### 3. Experiment Reproduction Steps (Brief)

1. **Environment Setup**
   - Python 3.9+
   - Recommended main dependencies (examples):
     - `torch`, `torchvision`
     - `numpy`, `pandas`, `scikit-learn`
     - `matplotlib`, `seaborn`
     - `optuna`
2. **Data Preparation**
   - **Generate SD Synthetic Data (Optional)**:
     - To generate SD datasets from scratch, follow these steps:
       1. **Generate Candidate Images**:
          ```bash
          python code/generate_sd15_dataset.py
          ```
          This will generate 2,500 candidate images (using SD v1.5, consistent with the paper).
       2. **Apply Three-Stage Filtering**:
          ```bash
          python code/three_stage_filter.py
          ```
          This will select 200 high-quality images from the 2,500 candidates (consistent with the paper).
     - Note: The filtering script requires additional dependencies: `opencv-python`, `transformers`, `scikit-learn`
   - Prepare Kvasir real data and SD-generated synthetic data according to the data paths in Section 2.
3. **Run Main Experiment Script**
   - Navigate to the directory
   - Run:
     - `python code/train_ablation_combo_full.py`
   - Complete 32 experiments and ablation analysis according to script parameters/configuration.

### 4. Reproduction Notes

- To fully reproduce from raw data:
  - Prepare raw Kvasir data and SD synthetic data;
  - Run training scripts and image generation scripts;

This README only covers the structure and reproduction steps related to the project itself, and does not involve other unrelated files. *** End Patch***
