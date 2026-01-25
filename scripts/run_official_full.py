"""
官方版全样本基准测试 (Official Full Benchmark)
1. Profiling: 50,000 样本 (训练集)
2. Testing: 10,000 样本 (测试集)
3. 算法: ICLR 2024 官方接口
"""
import sys
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime

from src.official_nac import OfficialNACWrapper
from src.loader import get_cifar10_loader, get_resnet18
from src.perturber import apply_ordered_perturbations
from src.visualizer import NACVisualizer
from src.nac_efficient import AnalysisResult

def main():
    print("="*60)
    print("OFFICIAL FULL BENCHMARK - 论文级全量运行")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = get_resnet18(pretrained=True).to(device)
    model.eval()
    
    analyzer = OfficialNACWrapper(model, device=device)
    
    # 获取层
    child_names = [n for n, _ in model.named_children()]
    target_layer = 'block3' if 'block3' in child_names else 'layer4'
    
    # 数据加载
    train_loader = get_cifar10_loader(batch_size=128, train=True)
    test_loader = get_cifar10_loader(batch_size=128, train=False)
    
    # 1. 全量 Profiling
    PROF_N = 50000
    print(f"\n[Step 1] Running Profile (N={PROF_N})...")
    analyzer.setup(train_loader, layer_names=[target_layer], valid_num=PROF_N)
    
    # 2. 实验定义
    experiments = {
        "clean": [],
        "gaussian_0.10": [('gaussian_noise', {'severity': 0.10})],
        "rotate_30": [('rotate', {'angle': 30})],
        "order_A_rotate_noise": [
            ('rotate', {'angle': 20}), 
            ('gaussian_noise', {'severity': 0.05})
        ],
        "order_B_noise_rotate": [
            ('gaussian_noise', {'severity': 0.05}), 
            ('rotate', {'angle': 20})
        ],
    }
    
    # 3. 全量测试
    TEST_N = 10000
    all_vis_results = {}
    print(f"\n[Step 2] Running Experiments (N={TEST_N})...")
    
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
            layer_name=target_layer,
            scores=scores_array,
            mean=float(scores_array.mean()),
            std=float(scores_array.std())
        )]
        print(f"    Mean: {all_vis_results[exp_name][0].mean:.4f}")

    # 4. 生成报告
    output_dir = 'official_full_benchmark'
    os.makedirs(output_dir, exist_ok=True)
    visualizer = NACVisualizer()
    visualizer.generate_full_report(all_vis_results, output_dir=output_dir, baseline_name='clean')
    
    print("\n[Done] All results in: official_full_benchmark/")

if __name__ == "__main__":
    main()
