
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from src.loader import get_resnet18, get_cifar10_loader
from src.perturber import Perturber
from src.official_nac import OfficialNACWrapper
from openood.evaluators.metrics import compute_all_metrics

# Configuration using strictly library APIs logic
DEBUG_MODE = False
LIMIT_SAMPLES = 128 if DEBUG_MODE else 10000
PROFILING_SAMPLES = 500 if DEBUG_MODE else 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_evaluation_cycle(model_name, model_alias, layers, test_loader, profiling_loader):
    print(f"\n" + "#"*80)
    print(f"PIPELINE: {model_alias} | LAYERS: {layers}")
    print("#"*80)
    
    # 1. Load Model (RobustBench API Wrapper)
    model = get_resnet18(model_name).to(DEVICE)
    model.eval()
    
    # 2. Init NAC & Profiling (OpenOOD API Wrapper)
    # Using 'valid_num' as per OpenOOD standard for profiling subset size
    analyzer = OfficialNACWrapper(model, device=DEVICE)
    analyzer.setup(profiling_loader, layer_names=layers, valid_num=PROFILING_SAMPLES)
    
    # 3. Official APS Search (OpenOOD Search Logic)
    # Using a high-energy Gaussian noise subset as OOD validation for parameter tuning
    # This aligns with the methodology of tuning for separation
    print("Running APS (Automatic Parameter Search)...")
    val_indices = range(64)
    val_subset = Subset(test_loader.dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=32)
    
    # Create OOD Validation Set using Advex-UAR API via Perturber
    perturber_val = Perturber(model, DEVICE)
    perturber_val.add('gaussian_noise', severity=0.1)
    
    ood_val_imgs = []
    for x, _ in val_loader:
        # Move to CPU immediately to save GPU memory during dataset construction
        ood_val_imgs.append(perturber_val.apply(x).cpu())
    
    # Helper dataset to pair images with dummy labels for DataLoader compatibility
    class ListDS(torch.utils.data.Dataset):
        def __init__(self, imgs): self.imgs = imgs
        def __len__(self): return len(self.imgs)
        def __getitem__(self, i): return self.imgs[i], 0
    
    ood_val_loader = DataLoader(ListDS(torch.cat(ood_val_imgs)), batch_size=32)
    # Trigger APS
    analyzer.run_aps(val_loader, ood_val_loader)
    
    # 4. Full Benchmark Execution
    # Define perturbations using standard naming conventions mapped in src/perturber.py
    perturbations = {
        "Clean": [],
        "AutoAttack_Linf": [('autoattack', {'norm': 'Linf', 'eps': 8/255, 'version': 'fast'})],
        "Gaussian": [('gaussian_noise', {'severity': 0.1})],
        # RotationAttack in advex-uar takes angle or angle_range
        "Rotation": [('rotate', {'angle': 30})],
        # Advex-UAR attacks (CIFAR-10 calibrated ranges in README / calibs.out)
        "Elastic": [('elastic', {'eps': 16.0, 'n_iters': 10})],
        "Fog": [('fog', {'eps': 256.0, 'n_iters': 10})],
        "Snow": [('snow', {'eps': 2.0, 'n_iters': 10})],
        "Gabor": [('gabor', {'eps': 25.0, 'n_iters': 10})],
    }

    # JPEG eps scan configs from advex-uar CIFAR-10 calibrations (README / calibs.out).
    # Use n_iters=200 to match the original calibration setting.
    jpeg_linf_eps = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]
    jpeg_l2_eps = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    jpeg_l1_eps = [2.0, 8.0, 64.0, 256.0, 512.0, 1024.0]

    for eps in jpeg_linf_eps:
        perturbations[f"JPEG_Linf_{eps}"] = [('jpeg', {'norm': 'linf', 'eps': eps, 'n_iters': 200})]
    for eps in jpeg_l2_eps:
        perturbations[f"JPEG_L2_{eps}"] = [('jpeg', {'norm': 'l2', 'eps': eps, 'n_iters': 200})]
    for eps in jpeg_l1_eps:
        perturbations[f"JPEG_L1_{eps}"] = [('jpeg', {'norm': 'l1', 'eps': eps, 'n_iters': 200})]
    
    results = []
    id_scores = None # To store clean scores for AUROC calculation
    perturber = Perturber(model, DEVICE)
    
    for p_name, p_configs in perturbations.items():
        perturber.clear()
        for name, params in p_configs: perturber.add(name, **params)
        
        all_scores, all_labels, all_preds = [], [], []
        
        for images, labels in tqdm(test_loader, desc=f"Testing {p_name}"):
            images = images.to(DEVICE)
            # Apply perturbation (Libraries handle scaling internally via Perturber wrapper)
            p_images = perturber.apply(images, labels.to(DEVICE))
            
            # NAC Inference (OpenOOD API)
            # NAC often requires gradients for certain calculations (like Gradient-based coverage)
            with torch.set_grad_enabled(True):
                scores = analyzer.score_batch(p_images)
                all_scores.append(scores.cpu().numpy())
            
            # Model Accuracy (Standard Inference)
            with torch.no_grad():
                outputs = model(p_images)
                all_preds.append(outputs.argmax(1).cpu().numpy())
                all_labels.append(labels.numpy())
                
        # Aggregate results
        scores_arr = np.concatenate(all_scores)
        preds_arr = np.concatenate(all_preds)
        labels_arr = np.concatenate(all_labels)
        
        acc = (preds_arr == labels_arr).mean()
        
        if p_name == "Clean":
            id_scores = scores_arr
            id_preds = preds_arr
            # Baseline metrics
            final_metrics = [0, 0.5, 0, 0, acc] 
        else:
            # Metric Calculation using strictly OpenOOD API
            # compute_all_metrics expects: (scores, labels, preds)
            # Labels convention: 1 for ID, -1 for OOD
            combined_scores = np.concatenate([id_scores, scores_arr])
            combined_labels = np.concatenate([np.ones_like(id_scores), -1 * np.ones_like(scores_arr)])
            
            # Combine predictions
            # id_preds needs to be captured from the clean run
            combined_preds = np.concatenate([id_preds, preds_arr])
            
            # Returns: [FPR@95, AUROC, AUPR_IN, AUPR_OUT, ACC]
            raw_metrics = compute_all_metrics(combined_scores, combined_labels, combined_preds)
            final_metrics = raw_metrics
            
        res = {
            "model": model_alias, "perturbation": p_name,
            "accuracy": acc, "nac_mean": scores_arr.mean(),
            "auroc": final_metrics[1], "fpr95": final_metrics[0]
        }
        results.append(res)
        print(f" -> Acc: {acc:.4f} | AUROC: {res['auroc']:.4f} | FPR95: {res['fpr95']:.4f}")
        
    return results

def main():
    # Load Data using RobustBench API logic (wrapped in loader.py)
    # Using 'n_examples' to limit size strictly as per RobustBench API
    test_loader = get_cifar10_loader(batch_size=32, n_examples=LIMIT_SAMPLES)
    profiling_loader = get_cifar10_loader(batch_size=128, n_examples=PROFILING_SAMPLES)
    
    # Models to test: Standard RB model vs Robust RB model (Gowal2021)
    # Layers chosen for ensemble based on ICLR paper recommendations (multi-depth)
    tasks = [
        ('Standard', 'ResNet18_Std', ['block1.layer.2', 'block2.layer.2', 'block3.layer.2']),
        ('Gowal2021Improving_28_10_ddpm_100m', 'WideResNet_Robust', ['layer.0.block.3', 'layer.1.block.3', 'layer.2.block.3'])
    ]
    
    all_res = []
    # Ensure output directory exists
    os.makedirs("total_benchmark_results", exist_ok=True)
    
    for m_name, m_alias, layers in tasks:
        try:
            all_res.extend(run_evaluation_cycle(m_name, m_alias, layers, test_loader, profiling_loader))
        except Exception as e:
            print(f"Error evaluating {m_alias}: {e}")
            import traceback
            traceback.print_exc()
        
    if all_res:
        df = pd.DataFrame(all_res)
        df.to_csv("total_benchmark_results/benchmark_results.csv", index=False)
        print("\nBenchmark Complete. Results saved to total_benchmark_results/benchmark_results.csv")
        print("Summary:")
        print(df[["model", "perturbation", "accuracy", "auroc", "fpr95"]])
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
