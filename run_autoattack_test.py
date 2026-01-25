"""
AutoAttack + NAC 专项实验 (诊断增强版)
验证对抗攻击对神经元激活覆盖率的影响
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm

from src.official_nac import OfficialNACWrapper
from src.loader import get_cifar10_loader, get_resnet18
from src.perturber import apply_ordered_perturbations

def run_exp(model, analyzer, test_loader, device, name, target_layer, resize=None):
    print(f"\n" + "="*40)
    print(f"RUNNING EXP: {name}")
    print("="*40)
    
    TEST_LIMIT = 16
    results = {}
    
    # Pre-normalization for standard ImageNet models
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Clean
    clean_scores = []
    n_count = 0
    for images, labels in test_loader:
        if n_count >= TEST_LIMIT: break
        x = images.to(device)
        if resize: x = F.interpolate(x, size=resize, mode='bilinear')
        if name.startswith("Standard"): x = normalize(x)
        
        batch_scores = analyzer.score_batch(x)
        clean_scores.append(batch_scores.cpu().numpy())
        n_count += images.shape[0]
    results['clean'] = np.concatenate(clean_scores)
    
    # Adv
    aa_scores = []
    correct_clean = 0
    correct_adv = 0
    n_count = 0
    for images, labels in test_loader:
        if n_count >= TEST_LIMIT: break
        images, labels = images.to(device), labels.to(device)
        
        x = images.clone()
        if resize: x = F.interpolate(x, size=resize, mode='bilinear')
        
        class NormalizedModel(nn.Module):
            def __init__(self, m, n):
                super().__init__(); self.m = m; self.n = n
            def forward(self, x): return self.m(self.n(x))
        
        attack_model = model
        if name.startswith("Standard"):
            attack_model = NormalizedModel(model, normalize)

        p_images = apply_ordered_perturbations(
            x.clone(), 
            [('autoattack', {'norm': 'Linf', 'eps': 32/255, 'version': 'standard', 'labels': labels, 'verbose': True})],
            model=attack_model,
            device=device
        )
        
        with torch.no_grad():
            clean_out = attack_model(x)
            adv_out = attack_model(p_images)
            correct_clean += (clean_out.argmax(1) == labels).sum().item()
            correct_adv += (adv_out.argmax(1) == labels).sum().item()

        batch_scores = analyzer.score_batch(p_images)
        aa_scores.append(batch_scores.cpu().numpy())
        n_count += images.shape[0]
        
    results['adv'] = np.concatenate(aa_scores)
    
    print(f"[{name}] Acc: {correct_clean/n_count:.1%} -> {correct_adv/n_count:.1%}")
    print(f"[{name}] NAC: {results['clean'].mean():.4f} -> {results['adv'].mean():.4f}")
    return results

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = get_cifar10_loader(batch_size=128, train=True)
    test_loader = get_cifar10_loader(batch_size=32, train=False)

    # 1. Standard ResNet18 (as control)
    from torchvision.models import resnet18, ResNet18_Weights
    print("\n[Step 0] Initializing Standard Model Control Group...")
    std_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    std_model.eval()
    std_analyzer = OfficialNACWrapper(std_model, device=device)
    std_analyzer.setup(train_loader, layer_names=['layer4'], valid_num=100)
    run_exp(std_model, std_analyzer, test_loader, device, "Standard-ResNet18", "layer4", resize=(224, 224))

    # 2. Robust WideResNet
    print("\n[Step 1] Initializing Robust Model Experimental Group...")
    robust_model = get_resnet18().to(device)
    robust_model.eval()
    robust_analyzer = OfficialNACWrapper(robust_model, device=device)
    robust_analyzer.setup(train_loader, layer_names=['layer.2'], valid_num=100)
    run_exp(robust_model, robust_analyzer, test_loader, device, "Robust-WideResNet", "layer.2")

if __name__ == "__main__":
    main()
