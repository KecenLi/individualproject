"""
第三阶段 - 官方接口版本
完全调用 ICLR 2024 (ood_coverage) 官方 NACPostprocessor 接口
"""
import sys
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime

# 核心：使用官方封装
from src.official_nac import OfficialNACWrapper
from src.loader import get_cifar10_loader, get_resnet18
from src.perturber import apply_ordered_perturbations
from src.visualizer import NACVisualizer
from src.nac_efficient import AnalysisResult

def main():
    print("="*60)
    print("PHASE 3: OFFICIAL INTERFACE VERSION")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 加载模型
    print("\n[Step 1] Loading Model...")
    model = get_resnet18(pretrained=True)
    model = model.to(device)
    model.eval()
    
    # 2. 准备官方分析器
    print("\n[Step 2] Initializing Official NAC Wrapper...")
    analyzer = OfficialNACWrapper(model, device=device)
    
    # 自动识别层名
    child_names = [n for n, _ in model.named_children()]
    # 官方推荐针对 CIFAR-10 的 ResNet 监控 avgpool，我们的 WideResNet 监控最后几个 block
    target_layer = 'block3' if 'block3' in child_names else 'layer4'
    print(f"Targeting layer: {target_layer}")
    
    # 3. 加载数据 (官方 Profiling 需要训练集)
    print("\n[Step 3] Loading Data for Profiling...")
    # 注意: 官方接口默认 valid_num=1000
    train_loader = get_cifar10_loader(batch_size=64, train=True)
    test_loader = get_cifar10_loader(batch_size=64, train=False)
    
    # 执行 Profiling
    print(f"\n[Running Official setup()...]")
    analyzer.setup(train_loader, layer_names=[target_layer])
    
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
    
    # 5. 运行全量测试
    TEST_SAMPLES = 2000
    all_vis_results = {}
    print(f"\n[Step 4] Running {len(experiments)} Experiments (N={TEST_SAMPLES})...")
    
    for exp_name, transforms in experiments.items():
        print(f"\n>>> Exp: {exp_name}")
        scores_list = []
        n_count = 0
        
        for images, labels in tqdm(test_loader, desc=exp_name):
            if n_count >= TEST_SAMPLES:
                break
                
            images = images.to(device)
            # 应用扰动
            if transforms:
                with torch.no_grad():
                    images = apply_ordered_perturbations(
                        images, transforms, model=model, device=device
                    )
            
            # 使用官方接口打分
            batch_scores = analyzer.score_batch(images)
            scores_list.extend(batch_scores.cpu().numpy().tolist())
            n_count += images.shape[0]
            
        scores_array = np.array(scores_list)
        mean_val = float(scores_array.mean())
        std_val = float(scores_array.std())
        
        all_vis_results[exp_name] = [AnalysisResult(
            perturbation_name=exp_name,
            layer_name=target_layer,
            scores=scores_array,
            mean=mean_val,
            std=std_val
        )]
        
        print(f"    NAC Score: {mean_val:.4f} (σ={std_val:.4f})")

    # 6. 生成报告
    output_dir = 'official_output'
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = NACVisualizer()
    visualizer.generate_full_report(all_vis_results, output_dir=output_dir, baseline_name='clean')
    
    print("\n" + "="*60)
    print(f"OFFICIAL VERSION COMPLETE. Results in: {output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()
