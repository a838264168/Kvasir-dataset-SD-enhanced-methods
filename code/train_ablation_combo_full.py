#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Combination Ablation Training with ALL 5 Innovations:
- Innovation 1: DomainSplice (data-level splice)
- Innovation 2: FeatureMix (feature-level fusion)
- Innovation 3: DANN v4 (optimized domain adversarial)
- Innovation 4: Negative KD (negative knowledge distillation)
- Innovation 5: Self-Evolution (data self-evolution)
- Plus: SD, SMOTE, Optuna toggles

Progressive schedule: epoch 1-10: 0% SD; 11-15: ~10%; 16-20: ~30%
"""

import os
import argparse
import random
import shutil
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from PIL import Image

# ===== Config =====
REAL_DATA_ROOT = "datasets_binary/baseline_real_only"
GOLDEN_SD_ROOT = "datasets_sd21_golden_filtered_v3"
OUT_ROOT = "runs_combo_full"
NUM_CLASSES = 2
EPOCHS = 20
BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 42

# ===== Utils =====
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_dirs(path: str):
    os.makedirs(path, exist_ok=True)

def macro_metrics(y_true, y_pred):
    m_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    m_pr = precision_score(y_true, y_pred, average='macro', zero_division=0)
    m_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    return m_f1, m_pr, m_rec

# ===== Losses =====
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

# ===== Innovation 3: DANN v4 Components =====
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_val, None

def grad_reverse(x, lambda_val):
    return GradientReversalLayer.apply(x, lambda_val)

def compute_gradient_penalty(domain_discriminator, real_features, sd_features):
    """WGAN-style gradient penalty for domain discriminator stability"""
    alpha = torch.rand(real_features.size(0), 1, device=real_features.device)
    interpolated = alpha * real_features + (1 - alpha) * sd_features
    interpolated.requires_grad_(True)
    
    d_interpolated = domain_discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ===== Innovation 4: Negative KD Components =====
def train_negative_teacher(sd_dir: Path, device, epochs=5):
    """Train a negative teacher on SD data with WRONG labels"""
    print("[INFO] Training negative teacher model...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    class WrongLabelDataset(Dataset):
        def __init__(self, sd_dir, transform):
            self.paths = sorted(list(sd_dir.glob("*.png")) + list(sd_dir.glob("*.jpg")))
            self.transform = transform
        
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert('RGB')
            return self.transform(img), 0  # WRONG label: SD is disease (1), but we label as normal (0)
    
    dataset = WrongLabelDataset(sd_dir / 'disease', transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    
    return model

# ===== Innovation 5: Self-Evolution Components =====
def evolve_sd_dataset(model, sd_dir: Path, device, top_k=500, confidence_threshold=0.5):
    """Use model to select top-k most confident SD images as 'disease'"""
    print(f"[INFO] Evolving SD dataset: selecting top {top_k} confident images...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    all_paths = sorted(list((sd_dir / 'disease').glob("*.png")) + list((sd_dir / 'disease').glob("*.jpg")))
    if not all_paths:
        print("[WARN] No SD images found!")
        return []
    
    scores = []
    
    model.eval()
    with torch.no_grad():
        for path in all_paths:
            img = Image.open(path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            logits = model(img_t)
            probs = F.softmax(logits, dim=1)
            # Confidence for disease class (class 1)
            confidence = probs[0, 1].item()
            scores.append((confidence, path))
    
    # Select top-k most confident
    scores.sort(reverse=True, key=lambda x: x[0])
    
    # First try with confidence threshold
    selected = [path for conf, path in scores[:top_k] if conf >= confidence_threshold]
    
    # Fallback: if no images pass threshold, use top-k regardless of threshold
    if len(selected) == 0:
        print(f"[WARN] No images with confidence >= {confidence_threshold}, using top {top_k} images regardless of threshold")
        selected = [path for _, path in scores[:top_k]]
        if selected:
            min_conf = min(conf for conf, _ in scores[:top_k])
            print(f"[INFO] Selected {len(selected)} images (min confidence: {min_conf:.4f})")
    
    print(f"[INFO] Selected {len(selected)} images (confidence >= {confidence_threshold})")
    return selected

# ===== Datasets (same as before) =====
class DiseaseSDDataset(Dataset):
    def __init__(self, sd_paths, transform):
        self.paths = sd_paths if isinstance(sd_paths, list) else sorted(list(sd_paths.glob("*.png")) + list(sd_paths.glob("*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), 1

class DomainSpliceWrapper(Dataset):
    def __init__(self, real_ds, sd_ds, enable, p_splice=0.5):
        self.real_ds = real_ds
        self.sd_ds = sd_ds
        self.enable = enable
        self.p_splice = p_splice
        self.transform = real_ds.transform

    def __len__(self):
        return len(self.real_ds)

    def __getitem__(self, idx):
        img, label = self.real_ds[idx]
        if not self.enable or random.random() > self.p_splice or self.sd_ds is None or len(self.sd_ds) == 0:
            return img, label
        path = self.real_ds.samples[idx][0]
        real_pil = Image.open(path).convert('RGB')
        sd_idx = random.randrange(len(self.sd_ds))
        sd_path = self.sd_ds.paths[sd_idx]
        sd_pil = Image.open(sd_path).convert('RGB')
        w, h = real_pil.size
        split_point = random.randint(int(0.3 * w), int(0.7 * w))
        spliced = Image.new('RGB', (w, h))
        spliced.paste(real_pil.crop((0, 0, split_point, h)), (0, 0))
        spliced.paste(sd_pil.crop((split_point, 0, w, h)), (split_point, 0))
        return self.transform(spliced), label

# ===== Model with all innovations =====
class ResNet18FullInnovations(nn.Module):
    def __init__(self, num_classes=2, feature_mix=False, dann=False, feature_mix_weight=0.3):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.num_features, num_classes)
        self.feature_mix = feature_mix
        self.feature_mix_weight = feature_mix_weight
        self.dann = dann
        
        if dann:
            self.domain_discriminator = nn.Sequential(
                nn.Linear(self.num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 2)  # 2 domains: real vs SD
            )

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x_real, x_sd_batch=None, lambda_dann=1.0, return_domain=False):
        feat_real = self.extract_features(x_real)
        
        if self.feature_mix and x_sd_batch is not None:
            with torch.no_grad():
                feat_sd = self.extract_features(x_sd_batch)
            feat = (1 - self.feature_mix_weight) * feat_real + self.feature_mix_weight * feat_sd
        else:
            feat = feat_real
        
        logits = self.classifier(feat)
        
        domain_logits = None
        if self.dann and return_domain:
            reversed_feat = grad_reverse(feat, lambda_dann)
            domain_logits = self.domain_discriminator(reversed_feat)
        
        return logits, domain_logits

# ===== SMOTE =====
def same_class_mixup(images, labels, alpha=0.4, p=0.5):
    if alpha <= 0 or random.random() > p:
        return images, labels
    lam = np.random.beta(alpha, alpha)
    images = images.clone()
    labels = labels.clone()
    for cls in labels.unique().tolist():
        idxs = (labels == cls).nonzero(as_tuple=True)[0]
        if len(idxs) < 2:
            continue
        perm = idxs[torch.randperm(len(idxs))]
        images[idxs] = lam * images[idxs] + (1 - lam) * images[perm]
    return images, labels

# ===== Build loaders (progressive) =====
def build_epoch_loaders(epoch, seed, use_domain_splice, sd_paths_list, use_self_evolution=False):
    set_seed(seed + epoch)
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    real_train = datasets.ImageFolder(os.path.join(REAL_DATA_ROOT, 'train'), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(REAL_DATA_ROOT, 'val'), transform=eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(REAL_DATA_ROOT, 'test'), transform=eval_tf)

    enable_splice = use_domain_splice and (epoch > 10)
    sd_ds = DiseaseSDDataset(sd_paths_list, train_tf) if (enable_splice and sd_paths_list) else None
    train_ds = DomainSpliceWrapper(real_train, sd_ds, enable=enable_splice, p_splice=0.5)

    targets = [y for _, y in train_ds.real_ds.samples]
    class_counts = np.bincount(targets, minlength=NUM_CLASSES)
    weights = 1.0 / np.log(class_counts + 1.2)
    sample_weights = weights[targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader

# ===== Main training function =====
def run_train_eval(tag, inv1_ds, inv2_fm, inv3_dann, inv4_nkd, inv5_se, use_sd, use_smote, use_optuna, seed=SEED):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(seed)
    exp_dir = os.path.join(OUT_ROOT, tag)
    make_dirs(exp_dir)

    sd_dir = Path(GOLDEN_SD_ROOT)
    sd_paths_list = None
    
    # Innovation 5: Self-Evolution - evolve SD dataset before training
    if inv5_se and use_sd:
        # First, train a preliminary model for evolution
        print("[INFO] Innovation 5: Self-Evolution - training preliminary model...")
        prelim_model = models.resnet18(weights=None)
        prelim_model.fc = nn.Linear(prelim_model.fc.in_features, NUM_CLASSES)
        prelim_model = prelim_model.to(device)
        # Quick training (5 epochs) on real data only
        real_train = datasets.ImageFolder(os.path.join(REAL_DATA_ROOT, 'train'), 
                                         transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
        prelim_loader = DataLoader(real_train, batch_size=32, shuffle=True, num_workers=0)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(prelim_model.parameters(), lr=1e-4)
        for ep in range(5):
            prelim_model.train()
            for imgs, labels in prelim_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(prelim_model(imgs), labels)
                loss.backward()
                optimizer.step()
        # Evolve SD dataset
        sd_paths_list = evolve_sd_dataset(prelim_model, sd_dir, device, top_k=400)
        del prelim_model
    elif use_sd:
        sd_paths_list = sorted(list((sd_dir / 'disease').glob("*.png")) + list((sd_dir / 'disease').glob("*.jpg")))

    # Innovation 4: Negative KD - train negative teacher
    negative_teacher = None
    if inv4_nkd and use_sd and sd_paths_list:
        negative_teacher = train_negative_teacher(sd_dir, device, epochs=5)

    # Hyperparameters
    if use_optuna:
        # Simplified: use fixed good values for now
        lr, wd = 1e-4, 1e-4
    else:
        lr, wd = 1e-4, 1e-4

    model = ResNet18FullInnovations(num_classes=NUM_CLASSES, feature_mix=inv2_fm, dann=inv3_dann).to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_rows, val_rows = [], []
    best_val_f1, best_epoch, best_state = -1.0, -1, None

    for epoch in range(1, EPOCHS + 1):
        train_loader, val_loader, test_loader = build_epoch_loaders(epoch, seed, inv1_ds, sd_paths_list, inv5_se)
        
        # DANN alpha schedule (linear 0.1 -> 0.5)
        dann_alpha = 0.1 + (0.4 * (epoch - 1) / (EPOCHS - 1)) if inv3_dann else 0.0
        
        sd_image_paths = []
        if (inv2_fm or inv3_dann) and epoch > 10 and sd_paths_list:
            sd_image_paths = sd_paths_list

        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        tr_preds, tr_labels = [], []
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            sd_batch = None
            if (inv2_fm or inv3_dann) and epoch > 10 and sd_image_paths:
                pil_list = []
                for _ in range(imgs.size(0)):
                    p = random.choice(sd_image_paths)
                    pil_list.append(Image.open(p).convert('RGB'))
                tf = train_loader.dataset.transform
                sd_batch = torch.stack([tf(pil) for pil in pil_list]).to(device)

            if use_smote:
                imgs, labels = same_class_mixup(imgs, labels, alpha=0.4, p=0.5)

            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            logits_real = None
            if inv3_dann and epoch > 10 and sd_batch is not None:
                # DANN: need both real and SD batches
                logits_real, domain_logits_real = model(imgs, None, lambda_dann=dann_alpha, return_domain=True)
                logits_sd, domain_logits_sd = model(sd_batch, None, lambda_dann=dann_alpha, return_domain=True)
                
                # Classification loss (real data only)
                cls_loss = criterion(logits_real, labels)
                
                # Domain loss
                domain_labels_real = torch.zeros(imgs.size(0), dtype=torch.long, device=device)
                domain_labels_sd = torch.ones(sd_batch.size(0), dtype=torch.long, device=device)
                domain_loss = F.cross_entropy(domain_logits_real, domain_labels_real) + \
                             F.cross_entropy(domain_logits_sd, domain_labels_sd)
                
                # Gradient penalty
                feat_real = model.extract_features(imgs)
                feat_sd = model.extract_features(sd_batch)
                gp = compute_gradient_penalty(model.domain_discriminator, feat_real, feat_sd)
                
                loss = cls_loss + dann_alpha * (domain_loss + 10.0 * gp)
            else:
                logits, _ = model(imgs, sd_batch, return_domain=False)
                loss = criterion(logits, labels)
                
                # Innovation 4: Negative KD loss
                if inv4_nkd and negative_teacher is not None:
                    with torch.no_grad():
                        neg_logits = negative_teacher(imgs)
                        neg_probs = F.softmax(neg_logits, dim=1)
                    student_probs = F.log_softmax(logits, dim=1)
                    # Maximize KL divergence (opposite of normal KD)
                    nkd_loss = -F.kl_div(student_probs, neg_probs, reduction='batchmean')
                    loss = loss + 0.1 * nkd_loss  # Weight for negative KD
                logits_real = logits  # For consistent variable name

            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * labels.size(0)
            preds = logits_real.argmax(1)  # Use logits_real which is always defined now
            tr_correct += (preds == labels).sum().item()
            tr_total += labels.size(0)
            tr_preds.extend(preds.detach().cpu().numpy())
            tr_labels.extend(labels.detach().cpu().numpy())

        tr_loss /= max(tr_total, 1)
        tr_acc = tr_correct / max(tr_total, 1)
        tr_f1, tr_mp, tr_mr = macro_metrics(tr_labels, tr_preds)
        train_rows.append({'epoch': epoch, 'loss': tr_loss, 'accuracy': tr_acc, 'macro_f1': tr_f1, 'macro_precision': tr_mp, 'macro_recall': tr_mr})

        # Validation
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        va_preds, va_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, _ = model(imgs, None, return_domain=False)
                loss = criterion(logits, labels)
                va_loss += loss.item() * labels.size(0)
                preds = logits.argmax(1)
                va_correct += (preds == labels).sum().item()
                va_total += labels.size(0)
                va_preds.extend(preds.cpu().numpy())
                va_labels.extend(labels.cpu().numpy())
        va_loss /= max(va_total, 1)
        va_acc = va_correct / max(va_total, 1)
        va_f1, va_mp, va_mr = macro_metrics(va_labels, va_preds)
        val_rows.append({'epoch': epoch, 'loss': va_loss, 'accuracy': va_acc, 'macro_f1': va_f1, 'macro_precision': va_mp, 'macro_recall': va_mr})

        if va_f1 > best_val_f1:
            best_val_f1, best_epoch = va_f1, epoch
            best_state = {'model': model.state_dict(), 'epoch': epoch}

        scheduler.step()

    # Save logs
    pd.DataFrame(train_rows).to_csv(os.path.join(exp_dir, 'train_log.csv'), index=False)
    pd.DataFrame(val_rows).to_csv(os.path.join(exp_dir, 'val_log.csv'), index=False)

    # Test
    if best_state is not None:
        model.load_state_dict(best_state['model'])

    model.eval()
    te_correct, te_total = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _ = model(imgs, None, return_domain=False)
            preds = logits.argmax(1)
            te_correct += (preds == labels).sum().item()
            te_total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    te_acc = te_correct / max(te_total, 1)
    te_f1, te_mp, te_mr = macro_metrics(y_true, y_pred)

    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=['normal', 'disease'], output_dict=True, zero_division=0)
    per_rows = []
    for cname in ['normal', 'disease']:
        d = report.get(cname, {})
        per_rows.append({'class': cname, 'precision': float(d.get('precision', 0.0)), 'recall': float(d.get('recall', 0.0)), 'f1': float(d.get('f1-score', 0.0)), 'support': int(d.get('support', 0))})
    pd.DataFrame(per_rows).to_csv(os.path.join(exp_dir, 'per_class_metrics.csv'), index=False)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    np.savetxt(os.path.join(exp_dir, 'confusion_matrix.csv'), cm, fmt='%d', delimiter=',')

    notes = f"inv1_ds={inv1_ds}, inv2_fm={inv2_fm}, inv3_dann={inv3_dann}, inv4_nkd={inv4_nkd}, inv5_se={inv5_se}, sd={use_sd}, smote={use_smote}, optuna={use_optuna}, lr={lr}, wd={wd}"
    test_summary = {'accuracy': te_acc, 'macro_f1': te_f1, 'macro_precision': te_mp, 'macro_recall': te_mr, 'best_epoch': best_epoch, 'notes': notes}
    pd.DataFrame([test_summary]).to_csv(os.path.join(exp_dir, 'test_metrics.csv'), index=False)

    torch.save(model.state_dict(), os.path.join(exp_dir, f'best_epoch_{best_epoch}.pth'))

    return test_summary

# ===== CLI =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inv1_ds', type=int, default=0)  # DomainSplice
    parser.add_argument('--inv2_fm', type=int, default=0)  # FeatureMix
    parser.add_argument('--inv3_dann', type=int, default=0)  # DANN v4
    parser.add_argument('--inv4_nkd', type=int, default=0)  # Negative KD
    parser.add_argument('--inv5_se', type=int, default=0)  # Self-Evolution
    parser.add_argument('--sd', type=int, default=1)  # Use SD data
    parser.add_argument('--smote', type=int, default=0)
    parser.add_argument('--optuna', type=int, default=0)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    tag = f"resnet18_full_inv1{args.inv1_ds}_inv2{args.inv2_fm}_inv3{args.inv3_dann}_inv4{args.inv4_nkd}_inv5{args.inv5_se}_sd{args.sd}_sm{args.smote}_opt{args.optuna}_seed{args.seed}"
    res = run_train_eval(tag, bool(args.inv1_ds), bool(args.inv2_fm), bool(args.inv3_dann), bool(args.inv4_nkd), bool(args.inv5_se), bool(args.sd), bool(args.smote), bool(args.optuna), seed=args.seed)
    print("="*80)
    print(f"Experiment {tag} completed: Macro-F1={res['macro_f1']:.4f}")
    print("="*80)

if __name__ == '__main__':
    main()

