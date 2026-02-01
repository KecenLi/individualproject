"""
Official ensemble + APS benchmark.
Uses multiple layers and APS to separate clean vs noise.
"""
import sys
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from src.official_nac import OfficialNACWrapper
from src.loader import get_cifar10_loader, get_resnet18
from src.perturber import apply_ordered_perturbations
from src.visualizer import NACVisualizer
from src.nac_efficient import AnalysisResult

def main():
    print("="*60)
    print("OFFICIAL ENSEMBLE & APS BENCHMARK")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model.
    model = get_resnet18(pretrained=True).to(device)
    model.eval()
    
    # Setup official wrapper.
    analyzer = OfficialNACWrapper(model, device=device)
    # Adjust layer names per model if needed.
    target_layers = ['layer.0', 'layer.1', 'layer.2']
    print(f"Ensembling Layers: {target_layers}")
    
    # Data loaders.
    train_loader = get_cifar10_loader(batch_size=128, train=True)
    test_loader = get_cifar10_loader(batch_size=128, train=False)
    
    # Profiling.
    print(f"\n[Step 1] Official Profiling (valid_num=1000)...")
    analyzer.setup(train_loader, layer_names=target_layers, valid_num=1000)
    
    # APS search.
    print(f"\n[Step 2] Official APS (Automatic Parameter Search)...")
    val_set = Subset(test_loader.dataset, range(128))
    id_val_loader = DataLoader(val_set, batch_size=64)
    
    class PerturbedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, device):
            self.dataset = dataset
            self.device = device
        def __len__(self): return len(self.dataset)
        def __getitem__(self, i):
            img, label = self.dataset[i]
            img = img.unsqueeze(0).to(self.device)
            p_img = apply_ordered_perturbations(img, [('gaussian_noise', {'severity': 0.1})], device=self.device)
            return p_images.squeeze(0).cpu(), label

    ood_val_loader = DataLoader(PerturbedDataset(val_set, device), batch_size=64)
    
    # Run APS.
    analyzer.run_aps(id_val_loader, ood_val_loader)
    
    # Benchmark experiments.
    experiments = {
        "clean": [],
        "gaussian_0.10": [('gaussian_noise', {'severity': 0.10})],
        "rotate_30": [('rotate', {'angle': 30})],
        "order_A_rotate_noise": [('rotate', {'angle': 20}), ('gaussian_noise', {'severity': 0.05})],
        "order_B_noise_rotate": [('gaussian_noise', {'severity': 0.05}), ('rotate', {'angle': 20})],
    }
    
    all_vis_results = {}
    print(f"\n[Step 3] Running Final Benchmark (N=10000)...")
    
    for exp_name, transforms in experiments.items():
        print(f"\n>>> Running: {exp_name}")
        scores_list = []
        for images, labels in tqdm(test_loader, desc=exp_name):
            images = images.to(device)
            if transforms:
                with torch.no_grad():
                    images = apply_ordered_perturbations(images, transforms, model=model, device=device)
            
            batch_scores = analyzer.score_batch(images)
            scores_list.extend(batch_scores.cpu().numpy().tolist())
            
        scores_array = np.array(scores_list)
        all_vis_results[exp_name] = [AnalysisResult(
            perturbation_name=exp_name,
            layer_name="Ensemble(b1+b2+b3)",
            scores=scores_array,
            mean=float(scores_array.mean()),
            std=float(scores_array.std())
        )]
    
    output_dir = 'official_aps_ensemble_output'
    os.makedirs(output_dir, exist_ok=True)
    visualizer = NACVisualizer()
    visualizer.generate_full_report(all_vis_results, output_dir=output_dir, baseline_name='clean')
    
    print(f"\nSUCCESS: Results saved to {output_dir}/")

if __name__ == "__main__":
    main()
