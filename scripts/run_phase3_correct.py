"""
第三阶段 - 正确高效版本

关键修复：
1. CIFAR-10使用原生32x32尺寸（不再resize到224x224）
2. 采用两阶段NAC策略（Profiling + Testing）
3. 简化代码逻辑，移除过度优化
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

LIBS_PATH = os.path.join(PROJECT_ROOT, 'libs')
if LIBS_PATH not in sys.path:
    sys.path.append(LIBS_PATH)

import torch
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

from src.loader import get_cifar10_loader, get_resnet18
from src.nac_efficient import EfficientNACAnalyzer, AnalysisResult
from src.perturber import apply_ordered_perturbations
from src.visualizer import NACVisualizer


# ============== 配置 ==============
BATCH_SIZE = int(os.environ.get('NAC_BATCH_SIZE', 64))         # 32x32图片可以用更大batch
PROFILE_SAMPLES = int(os.environ.get('NAC_PROFILE_SAMPLES', 1000))  # Profiling阶段样本数
TEST_SAMPLES = int(os.environ.get('NAC_TEST_SAMPLES', 2000))        # 每个实验的测试样本数
AA_VERSION = os.environ.get('AA_VERSION', 'standard')
ADVEX_ITERS = int(os.environ.get('ADVEX_ITERS', 20))


def main():
    print("="*60)
    print("第三阶段：正确高效版本")
    print("="*60)
    
    # 1. 设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 2. 加载模型
    print("\n加载模型...")
    model = get_resnet18()
    model = model.to(device)
    model.eval()
    
    # 检测层名
    child_names = [n for n, _ in model.named_children()]
    target_layer = 'block3' if 'block3' in child_names else 'layer4'
    print(f"Target layer: {target_layer}")
    print(f"Model architecture: {child_names}")
    
    # 3. 加载数据
    print(f"\n加载数据 (batch_size={BATCH_SIZE})...")
    # Profiling 必须用训练集 (id_loader_dict['main_train'])
    train_loader = get_cifar10_loader(batch_size=BATCH_SIZE, train=True)
    # 测试实验用测试集
    test_loader = get_cifar10_loader(batch_size=BATCH_SIZE, train=False)
    
    # 验证数据尺寸
    sample, _ = next(iter(train_loader))
    print(f"数据尺寸: {sample.shape} (应该是 [B, 3, 32, 32])")
    
    if sample.shape[-1] != 32:
        print("警告：数据尺寸不是32x32！")
        return
    
    # 4. 创建NAC分析器并Profiling
    print(f"\n[阶段1] NAC Profiling ({PROFILE_SAMPLES} 训练样本)...")
    # 使用论文中对齐的参数 (已在EfficientNACAnalyzer默认值中设置)
    analyzer = EfficientNACAnalyzer(model, [target_layer], device)
    # 使用训练集进行分布建立
    analyzer.profile(train_loader, max_samples=PROFILE_SAMPLES)
    
    # 5. 定义实验
    experiments = {
        "clean": [],
        "gaussian_0.05": [('gaussian_noise', {'severity': 0.05})],
        "gaussian_0.10": [('gaussian_noise', {'severity': 0.10})],
        "rotate_15": [('rotate', {'angle': 15})],
        "rotate_30": [('rotate', {'angle': 30})],
        # DeepG-style geometric transformations (lightweight ranges)
        "deepg_rotate_range": [('rotate', {'angle_range': (-5, 5)})],
        "deepg_translate_2px": [('translation', {'max_dx': 2, 'max_dy': 2})],
        "deepg_scale_range": [('scale', {'scale_range': (0.95, 1.05)})],
        "deepg_shear_range": [('shear', {'shear_x_range': (-5, 5)})],
        # AutoAttack (Lp-norm attacks)
        "autoattack_linf": [('autoattack', {'norm': 'Linf', 'eps': 8/255, 'version': AA_VERSION})],
        "autoattack_l2": [('autoattack', {'norm': 'L2', 'eps': 0.5, 'version': AA_VERSION})],
        # Advex-UAR (diverse attacks)
        "advex_elastic": [('elastic', {'eps': 2.0, 'n_iters': ADVEX_ITERS})],
        "advex_fog": [('fog', {'eps': 255.0, 'n_iters': ADVEX_ITERS})],
        "advex_snow": [('snow', {'eps': 2.0, 'n_iters': ADVEX_ITERS})],
        "advex_gabor": [('gabor', {'eps': 25.0, 'n_iters': ADVEX_ITERS})],
        "advex_jpeg_linf": [('jpeg', {'norm': 'linf', 'eps': 0.25, 'n_iters': ADVEX_ITERS})],
        "advex_jpeg_l2": [('jpeg', {'norm': 'l2', 'eps': 16.0, 'n_iters': ADVEX_ITERS})],
        "advex_jpeg_l1": [('jpeg', {'norm': 'l1', 'eps': 256.0, 'n_iters': ADVEX_ITERS})],
        "brightness_1.3": [('brightness', {'factor': 1.3})],
        "contrast_1.5": [('contrast', {'factor': 1.5})],
        # Order effects within geometric transforms
        "order_geom_A_rotate_translate": [
            ('rotate', {'angle': 15}),
            ('translation', {'dx': 2, 'dy': -1}),
        ],
        "order_geom_B_translate_rotate": [
            ('translation', {'dx': 2, 'dy': -1}),
            ('rotate', {'angle': 15}),
        ],
        # Order effects across geometric + noise
        "order_mix_A_rotate_noise": [
            ('rotate', {'angle': 20}),
            ('gaussian_noise', {'severity': 0.05})
        ],
        "order_mix_B_noise_rotate": [
            ('gaussian_noise', {'severity': 0.05}),
            ('rotate', {'angle': 20})
        ],
        "order_A_rotate_noise": [
            ('rotate', {'angle': 20}), 
            ('gaussian_noise', {'severity': 0.05})
        ],
        "order_B_noise_rotate": [
            ('gaussian_noise', {'severity': 0.05}), 
            ('rotate', {'angle': 20})
        ],
    }
    
    print(f"\n[阶段2] 运行 {len(experiments)} 个实验 ({TEST_SAMPLES} 样本/实验)...")
    
    # 6. 运行实验
    results = {}
    
    for exp_name, transforms in experiments.items():
        print(f"\n{'='*40}")
        print(f"实验: {exp_name}")
        
        all_scores = []
        n_samples = 0
        correct = 0
        seen = 0
        
        for images, labels in tqdm(test_loader, desc=exp_name):
            if n_samples >= TEST_SAMPLES:
                break
            
            images = images.to(device)
            labels = labels.to(device)

            remaining = TEST_SAMPLES - n_samples
            if images.shape[0] > remaining:
                images = images[:remaining]
                labels = labels[:remaining]
            
            # 应用扰动
            if transforms:
                images = apply_ordered_perturbations(
                    images, transforms, model=model, device=device, labels=labels
                )
            
            # 计算NAC分数
            scores = analyzer.score_batch(images)
            all_scores.extend(scores[target_layer].cpu().numpy().tolist())

            # 同步记录分类准确率（用于相关性分析）
            with torch.no_grad():
                preds = model(images).argmax(1)
                correct += (preds == labels).sum().item()
                seen += labels.shape[0]
            
            n_samples += images.shape[0]
            torch.cuda.empty_cache()
        
        scores_array = np.array(all_scores)
        accuracy = float(correct / max(seen, 1))
        results[exp_name] = {
            'scores': scores_array,
            'mean': float(scores_array.mean()),
            'std': float(scores_array.std()),
            'n_samples': len(scores_array),
            'accuracy': accuracy,
        }
        
        print(
            f"  Mean: {results[exp_name]['mean']:.4f}, "
            f"Std: {results[exp_name]['std']:.4f}, "
            f"Acc: {results[exp_name]['accuracy']:.4f}"
        )
    
    # 7. 对比结果
    print("\n" + "="*60)
    print("实验结果对比")
    print("="*60)
    
    baseline = results['clean']['mean']
    baseline_acc = results['clean']['accuracy']
    print(f"基线 (clean): {baseline:.4f}")
    print()

    # 计算 NAC 与准确率的相关性（跨扰动类型）
    corr_names = [k for k in results.keys() if k != 'clean']
    nac_means = np.array([results[k]['mean'] for k in corr_names], dtype=float)
    acc_vals = np.array([results[k]['accuracy'] for k in corr_names], dtype=float)
    if len(corr_names) > 1 and nac_means.std() > 0 and acc_vals.std() > 0:
        nac_acc_corr = float(np.corrcoef(nac_means, acc_vals)[0, 1])
    else:
        nac_acc_corr = 0.0
    print(f"NAC-Acc 相关系数 (excluding clean): {nac_acc_corr:+.4f}")
    
    for exp_name, data in results.items():
        if exp_name == 'clean':
            continue
        delta = data['mean'] - baseline
        acc_delta = data['accuracy'] - baseline_acc
        print(
            f"  {exp_name}: {data['mean']:.4f} (Δ = {delta:+.4f}) | "
            f"Acc: {data['accuracy']:.4f} (Δ = {acc_delta:+.4f})"
        )
    
    # 8. 保存结果
    os.makedirs('phase3_output', exist_ok=True)
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'batch_size': BATCH_SIZE,
            'profile_samples': PROFILE_SAMPLES,
            'test_samples': TEST_SAMPLES,
            'target_layer': target_layer,
            'input_size': '32x32',
            'aa_version': AA_VERSION,
            'advex_iters': ADVEX_ITERS,
        },
        'metrics': {
            'nac_acc_corr_excluding_clean': nac_acc_corr,
        },
        'results': {
            k: {
                'mean': v['mean'],
                'std': v['std'],
                'n_samples': v['n_samples'],
                'accuracy': v['accuracy'],
            }
            for k, v in results.items()
        }
    }
    
    with open('phase3_output/results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存到 phase3_output/results.json")
    
    # 9. 生成可视化
    print("\n生成可视化...")
    
    vis_results = {}
    for exp_name, data in results.items():
        vis_results[exp_name] = [AnalysisResult(
            perturbation_name=exp_name,
            layer_name=target_layer,
            scores=data['scores'],
            mean=data['mean'],
            std=data['std']
        )]
    
    visualizer = NACVisualizer()
    visualizer.generate_full_report(vis_results, output_dir='phase3_output', baseline_name='clean')
    
    print("\n" + "="*60)
    print("完成！输出目录: phase3_output/")
    print("="*60)


if __name__ == "__main__":
    main()
