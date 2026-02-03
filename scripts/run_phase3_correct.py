"""
Phase 3 baseline pipeline.
Uses native 32x32 CIFAR-10 and a profiling/testing split.
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


# Config.
BATCH_SIZE = int(os.environ.get('NAC_BATCH_SIZE', 64))
PROFILE_SAMPLES = int(os.environ.get('NAC_PROFILE_SAMPLES', 1000))
TEST_SAMPLES = int(os.environ.get('NAC_TEST_SAMPLES', 2000))
AA_VERSION = os.environ.get('AA_VERSION', 'standard')
ADVEX_ITERS = int(os.environ.get('ADVEX_ITERS', 20))
OUTPUT_DIR = os.environ.get('PHASE3_OUTPUT_DIR', 'phase3_output')


def main():
    print("="*60)
    print("第三阶段：正确高效版本")
    print("="*60)
    
    # Device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Model.
    print("\n加载模型...")
    model = get_resnet18()
    model = model.to(device)
    model.eval()
    
    # Target layer.
    child_names = [n for n, _ in model.named_children()]
    target_layer = 'block3' if 'block3' in child_names else 'layer4'
    print(f"Target layer: {target_layer}")
    print(f"Model architecture: {child_names}")
    
    # Data loaders.
    print(f"\n加载数据 (batch_size={BATCH_SIZE})...")
    # Profiling uses the train split.
    train_loader = get_cifar10_loader(batch_size=BATCH_SIZE, train=True)
    # Testing uses the test split.
    test_loader = get_cifar10_loader(batch_size=BATCH_SIZE, train=False)
    
    # Basic shape check.
    sample, _ = next(iter(train_loader))
    print(f"数据尺寸: {sample.shape} (应该是 [B, 3, 32, 32])")
    
    if sample.shape[-1] != 32:
        print("警告：数据尺寸不是32x32！")
        return
    
    # Profiling.
    print(f"\n[阶段1] NAC Profiling ({PROFILE_SAMPLES} 训练样本)...")
    # Analyzer uses paper-aligned defaults.
    analyzer = EfficientNACAnalyzer(model, [target_layer], device)
    # Build profile on the train split.
    analyzer.profile(train_loader, max_samples=PROFILE_SAMPLES)
    
    # Experiment set.
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

    # Optional experiment filter (comma-separated list). Example:
    #   EXP_SET="clean,gaussian_0.05,rotate_15"
    exp_set = os.environ.get('EXP_SET', '').strip()
    if exp_set and exp_set.lower() not in ('all', '*'):
        requested = [x.strip() for x in exp_set.split(',') if x.strip()]
        if requested:
            # Ensure clean is always included for baseline computations.
            if 'clean' not in requested:
                requested = ['clean'] + requested
            unknown = [x for x in requested if x not in experiments]
            if unknown:
                print(f"[WARN] Unknown experiments in EXP_SET will be ignored: {unknown}")
            filtered = {}
            for name in requested:
                if name in experiments:
                    filtered[name] = experiments[name]
            if not filtered:
                print("[ERROR] EXP_SET filtered out all experiments. Check names.")
                return
            experiments = filtered
    
    print(f"\n[阶段2] 运行 {len(experiments)} 个实验 ({TEST_SAMPLES} 样本/实验)...")
    
    # Run experiments.
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
            
            # Apply perturbations.
            if transforms:
                images = apply_ordered_perturbations(
                    images, transforms, model=model, device=device, labels=labels
                )
            
            # Score NAC.
            scores = analyzer.score_batch(images)
            all_scores.extend(scores[target_layer].cpu().numpy().tolist())

            # Track accuracy for correlation.
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
    
    # Aggregate and compare.
    print("\n" + "="*60)
    print("实验结果对比")
    print("="*60)
    
    baseline = results['clean']['mean']
    baseline_acc = results['clean']['accuracy']
    print(f"基线 (clean): {baseline:.4f}")
    print()

    # Correlation between NAC and accuracy (excluding clean).
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
    
    # Save results.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
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
    
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存到 {OUTPUT_DIR}/results.json")
    
    # Visualize.
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
    visualizer.generate_full_report(vis_results, output_dir=OUTPUT_DIR, baseline_name='clean')
    
    print("\n" + "="*60)
    print(f"完成！输出目录: {OUTPUT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
