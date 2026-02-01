"""
Full NAC benchmark.
Profiling: 50k train. Testing: 10k test.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'libs'))

import torch
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

from src.loader import get_cifar10_loader, get_resnet18
from src.nac_efficient import EfficientNACAnalyzer, AnalysisResult
from src.perturber import apply_ordered_perturbations
from src.visualizer import NACVisualizer


# Full-run defaults.
BATCH_SIZE = 128
PROFILE_SAMPLES = 50000
TEST_SAMPLES = 10000
OUTPUT_DIR = 'full_benchmark_output'

def main():
    print("="*60)
    print("NAC FULL BENCHMARK - 论文级全样本测试")
    print(f"Profiling: {PROFILE_SAMPLES} | Testing: {TEST_SAMPLES}")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model.
    print("\n[Step 1] 加载 RobustBench 模型...")
    model = get_resnet18(pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Target layer.
    child_names = [n for n, _ in model.named_children()]
    target_layer = 'block3' if 'block3' in child_names else 'layer4'
    print(f"Target Layer: {target_layer}")
    
    # Data loaders.
    print(f"\n[Step 2] 准备 DataLoader (Batch Size: {BATCH_SIZE})...")
    train_loader = get_cifar10_loader(batch_size=BATCH_SIZE, train=True)
    test_loader = get_cifar10_loader(batch_size=BATCH_SIZE, train=False)
    
    # Profiling.
    print(f"\n[Step 3] 执行全量 Profiling ({PROFILE_SAMPLES} 样本)...")
    analyzer = EfficientNACAnalyzer(model, [target_layer], device)
    analyzer.profile(train_loader, max_samples=PROFILE_SAMPLES)
    
    # Experiment set.
    experiments = {
        "clean": [],
        "gaussian_0.05": [('gaussian_noise', {'severity': 0.05})],
        "gaussian_0.10": [('gaussian_noise', {'severity': 0.10})],
        "rotate_30": [('rotate', {'angle': 30})],
        "brightness_1.3": [('brightness', {'factor': 1.3})],
        "order_A_rotate_noise": [
            ('rotate', {'angle': 20}), 
            ('gaussian_noise', {'severity': 0.05})
        ],
        "order_B_noise_rotate": [
            ('gaussian_noise', {'severity': 0.05}), 
            ('rotate', {'angle': 20})
        ],
    }
    
    # Run experiments.
    all_vis_results = {}
    final_stats = {}
    
    print(f"\n[Step 4] 开始全量测试 ({len(experiments)} 个实验)...")
    
    for exp_name, transforms in experiments.items():
        print(f"\n>>> Running: {exp_name}")
        scores_list = []
        
        for images, labels in tqdm(test_loader, desc=exp_name):
            images = images.to(device)
            
            # Apply perturbations.
            if transforms:
                with torch.no_grad():
                    images = apply_ordered_perturbations(
                        images, transforms, model=model, device=device
                    )
            
            # Score batch.
            batch_scores = analyzer.score_batch(images)
            scores_list.extend(batch_scores[target_layer].cpu().numpy().tolist())
            
        # Aggregate stats.
        scores_array = np.array(scores_list)
        mean_val = float(scores_array.mean())
        std_val = float(scores_array.std())
        
        final_stats[exp_name] = {
            'mean': mean_val,
            'std': std_val,
            'n_samples': len(scores_array)
        }
        
        all_vis_results[exp_name] = [AnalysisResult(
            perturbation_name=exp_name,
            layer_name=target_layer,
            scores=scores_array,
            mean=mean_val,
            std=std_val
        )]
        
        print(f"    Result: Mean={mean_val:.4f}, Std={std_val:.4f}")

    # Save and visualize.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save JSON.
    summary = {
        'timestamp': datetime.now().isoformat(),
        'settings': {
            'profile_samples': PROFILE_SAMPLES,
            'test_samples': TEST_SAMPLES,
            'target_layer': target_layer
        },
        'results': final_stats
    }
    with open(f'{OUTPUT_DIR}/full_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate plots.
    print(f"\n[Step 5] 正在生成可视化报表...")
    visualizer = NACVisualizer()
    visualizer.generate_full_report(all_vis_results, output_dir=OUTPUT_DIR, baseline_name='clean')
    
    print(f"\n{'='*60}")
    print(f"全样本基准测试完成！")
    print(f"数据已保存在: {OUTPUT_DIR}/")
    print("="*60)

if __name__ == "__main__":
    main()
