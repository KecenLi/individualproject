"""
NAC 全样本基准测试 (Full Benchmark)
该脚本旨在复现论文级别的统计精度：
1. Profiling: 使用完整的 CIFAR-10 训练集 (50,000 样本)
2. Testing: 使用完整的 CIFAR-10 测试集 (10,000 样本)
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


# ============== 全样本配置 ==============
BATCH_SIZE = 128         # 提高 Batch Size 以加快速度
PROFILE_SAMPLES = 50000  # 覆盖整个训练集
TEST_SAMPLES = 10000     # 覆盖整个测试集
OUTPUT_DIR = 'full_benchmark_output'

def main():
    print("="*60)
    print("NAC FULL BENCHMARK - 论文级全样本测试")
    print(f"Profiling: {PROFILE_SAMPLES} | Testing: {TEST_SAMPLES}")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 加载模型
    print("\n[Step 1] 加载 RobustBench 模型...")
    model = get_resnet18(pretrained=True)
    model = model.to(device)
    model.eval()
    
    # 自动识别层
    child_names = [n for n, _ in model.named_children()]
    target_layer = 'block3' if 'block3' in child_names else 'layer4'
    print(f"Target Layer: {target_layer}")
    
    # 2. 加载数据
    print(f"\n[Step 2] 准备 DataLoader (Batch Size: {BATCH_SIZE})...")
    train_loader = get_cifar10_loader(batch_size=BATCH_SIZE, train=True)
    test_loader = get_cifar10_loader(batch_size=BATCH_SIZE, train=False)
    
    # 3. 创建分析器并执行 Profiling
    print(f"\n[Step 3] 执行全量 Profiling ({PROFILE_SAMPLES} 样本)...")
    analyzer = EfficientNACAnalyzer(model, [target_layer], device)
    analyzer.profile(train_loader, max_samples=PROFILE_SAMPLES)
    
    # 4. 定义实验组
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
    
    # 5. 循环运行实验
    all_vis_results = {}
    final_stats = {}
    
    print(f"\n[Step 4] 开始全量测试 ({len(experiments)} 个实验)...")
    
    for exp_name, transforms in experiments.items():
        print(f"\n>>> Running: {exp_name}")
        scores_list = []
        
        for images, labels in tqdm(test_loader, desc=exp_name):
            images = images.to(device)
            
            # 应用扰动
            if transforms:
                with torch.no_grad():
                    images = apply_ordered_perturbations(
                        images, transforms, model=model, device=device
                    )
            
            # 计算分数
            batch_scores = analyzer.score_batch(images)
            scores_list.extend(batch_scores[target_layer].cpu().numpy().tolist())
            
        # 统计
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

    # 6. 保存与可视化
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存 JSON
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
    
    # 生成图表
    print(f"\n[Step 5] 正在生成可视化报表...")
    visualizer = NACVisualizer()
    visualizer.generate_full_report(all_vis_results, output_dir=OUTPUT_DIR, baseline_name='clean')
    
    print(f"\n{'='*60}")
    print(f"全样本基准测试完成！")
    print(f"数据已保存在: {OUTPUT_DIR}/")
    print("="*60)

if __name__ == "__main__":
    main()
