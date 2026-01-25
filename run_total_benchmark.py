
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

import sys
sys.path.append(os.path.join(os.getcwd(), 'OODRobustBench'))
from src.loader import get_resnet18, get_cifar10_loader
from src.perturber import Perturber
from src.official_nac import OfficialNACWrapper
from openood.evaluators.metrics import compute_all_metrics
from robustbench.data import CORRUPTION_DATASET_LOADERS, get_preprocessing
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

# Configuration using strictly library APIs logic
DEBUG_MODE = True
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
        "Gaussian": [('gaussian_noise', {'severity': 0.1})],
        "Rotation": [('rotate', {'angle': 30})],
        # AutoAttack - Standard Benchmark Settings
        "AutoAttack_Linf": [('autoattack', {'norm': 'Linf', 'eps': 8/255, 'version': 'standard'})],
        "AutoAttack_L2": [('autoattack', {'norm': 'L2', 'eps': 0.5, 'version': 'standard'})],
        
        # Advex-UAR Attacks - Using Calibrated Deltas (Epsilon 4/5 logic for strong attack)
        # Elastic: CIFAR-10 Calibrated Eps 5 -> 2.0. (Paper uses 100 iters)
        "Elastic": [('elastic', {'eps': 2.0, 'n_iters': 100})],
        
        # Fog: No CIFAR-10 calibration. Using ImageNet Eps 3 (512.0) conservatively.
        # Eps 4 is 2048 which might be too heavy for 32x32.
        "Fog": [('fog', {'eps': 512.0, 'n_iters': 100})],
        
        # Snow: No CIFAR-10 calibration. Using ImageNet Eps 4 (2.0). 
        # Snowflakes are additive, 2.0 intensity is visible but not overwhelming.
        "Snow": [('snow', {'eps': 2.0, 'n_iters': 100})],
        
        # Gabor: No CIFAR-10 calibration. Using ImageNet Eps 3 (25.0).
        # Eps 4 is 400, which is large. 
        "Gabor": [('gabor', {'eps': 25.0, 'n_iters': 100})],
        
        # JPEG: CIFAR-10 Calibrated
        # Linf Eps 4 -> 0.25 (range [0,1] effectively, but eps passed as is)
        "JPEG_Linf": [('jpeg', {'norm': 'linf', 'eps': 0.25, 'n_iters': 100})],
        # L2: No CIFAR calib. ImageNet Eps 2 -> 16.0 (Scaled down for smaller dimension)
        "JPEG_L2": [('jpeg', {'norm': 'l2', 'eps': 16.0, 'n_iters': 100})],
        # L1: CIFAR-10 Calibrated Eps 4 -> 256.0
        "JPEG_L1": [('jpeg', {'norm': 'l1', 'eps': 256.0, 'n_iters': 100})],
    }
    
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
            
            final_metrics = raw_metrics
            
        res = {
            "model": model_alias, "perturbation": p_name,
            "accuracy": acc, "nac_mean": scores_arr.mean(),
            "auroc": final_metrics[1], "fpr95": final_metrics[0]
        }
        results.append(res)
        print(f" -> Acc: {acc:.4f} | AUROC: {res['auroc']:.4f} | FPR95: {res['fpr95']:.4f}")
        
    # 5. OODRobustBench Corruptions Loop
    # We strictly follow the logic in oodrobustbench/eval.py
    # Loaders: CORRUPTION_DATASET_LOADERS[dataset][model](n, severity, dir, shuffle, [corr], prepr)
    
    corruptions_to_test = ['snow', 'fog', 'pixelate', 'jpeg_compression']
    # Use standard preprocessing for the model
    prepr = get_preprocessing(BenchmarkDataset.cifar_10, ThreatModel.Linf, model_name, None)
    
    # We test only Severity 5 for benchmark speed, or loop 1-5 if needed
    severity_levels = [5] 
    
    for c_name in corruptions_to_test:
        for sev in severity_levels:
            full_c_name = f"{c_name}_s{sev}"
            print(f"Testing Corruption: {full_c_name}")
            
            # Load corrupted dataset (downloads automatically to ./data)
            x_corr, y_corr = CORRUPTION_DATASET_LOADERS[BenchmarkDataset.cifar_10][ThreatModel.corruptions](
                LIMIT_SAMPLES, sev, "./data", False, [c_name], prepr
            )
            
            # Create Loader
            # y_corr might be a tensor already
            class TensorDS(torch.utils.data.Dataset):
                def __init__(self, x, y): self.x, self.y = x, y
                def __len__(self): return len(self.x)
                def __getitem__(self, i): return self.x[i], self.y[i]
                
            c_loader = DataLoader(TensorDS(x_corr, y_corr), batch_size=32, shuffle=False)
            
            all_scores, all_labels, all_preds = [], [], []
            
            for images, labels in tqdm(c_loader, desc=f"Eval {full_c_name}"):
                images = images.to(DEVICE)
                
                # NAC Inference
                with torch.set_grad_enabled(True):
                    scores = analyzer.score_batch(images)
                    all_scores.append(scores.cpu().numpy())

                # Model Inference
                with torch.no_grad():
                    outputs = model(images)
                    all_preds.append(outputs.argmax(1).cpu().numpy())
                    all_labels.append(labels.numpy())
            
            # Metrics
            scores_arr = np.concatenate(all_scores)
            preds_arr = np.concatenate(all_preds)
            labels_arr = np.concatenate(all_labels)
            acc = (preds_arr == labels_arr).mean()
            
            # Compute OOD metrics against Clean Baseline (id_scores)
            combined_scores = np.concatenate([id_scores, scores_arr])
            combined_labels = np.concatenate([np.ones_like(id_scores), -1 * np.ones_like(scores_arr)])
            combined_preds = np.concatenate([id_preds, preds_arr])
            
            raw_metrics = compute_all_metrics(combined_scores, combined_labels, combined_preds)
            
            res = {
                "model": model_alias, "perturbation": f"Corruption_{full_c_name}",
                "accuracy": acc, "nac_mean": scores_arr.mean(),
                "auroc": raw_metrics[1], "fpr95": raw_metrics[0]
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
