"""
可视化模块：生成PDF直方图和统计分析
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from pathlib import Path
import json

from .nac_efficient import AnalysisResult


class NACVisualizer:
    """NAC结果可视化器"""
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """初始化可视化器"""
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_coverage_histogram(self, 
                                results: Dict[str, List[AnalysisResult]],
                                layer_name: str,
                                save_path: Optional[str] = None,
                                bins: int = 50):
        """
        绘制覆盖率直方图（PDF）
        
        Args:
            results: {experiment_name: [AnalysisResult]}
            layer_name: 层名称
            save_path: 保存路径
            bins: 直方图bins数量
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for exp_name, exp_results in results.items():
            # 找到对应层的结果
            layer_result = next((r for r in exp_results if r.layer_name == layer_name), None)
            if layer_result is None:
                continue
            
            # 绘制直方图（归一化为概率密度）
            ax.hist(layer_result.scores, bins=bins, alpha=0.6, 
                   label=f"{exp_name} (μ={layer_result.mean:.3f})",
                   density=True, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('NAC Score', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'NAC Coverage Profile - {layer_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_comparison_bars(self,
                            results: Dict[str, List[AnalysisResult]],
                            baseline_name: str = 'clean',
                            save_path: Optional[str] = None):
        """
        绘制不同实验的NAC分数对比柱状图
        
        Args:
            results: {experiment_name: [AnalysisResult]}
            baseline_name: 基线实验名称
            save_path: 保存路径
        """
        # 收集所有层的数据
        layers = list(set(r.layer_name for exp_results in results.values() for r in exp_results))
        exp_names = list(results.keys())
        
        # 准备数据
        data = {layer: [] for layer in layers}
        for layer in layers:
            for exp_name in exp_names:
                exp_results = results[exp_name]
                layer_result = next((r for r in exp_results if r.layer_name == layer), None)
                if layer_result:
                    data[layer].append(layer_result.mean)
                else:
                    data[layer].append(0)
        
        # 绘图
        fig, axes = plt.subplots(1, len(layers), figsize=(6*len(layers), 5))
        if len(layers) == 1:
            axes = [axes]
        
        for idx, layer in enumerate(layers):
            ax = axes[idx]
            x = np.arange(len(exp_names))
            bars = ax.bar(x, data[layer], color=sns.color_palette("husl", len(exp_names)))
            
            # 标注数值
            for i, (bar, val) in enumerate(zip(bars, data[layer])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=9)
            
            ax.set_xticks(x)
            ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel('Mean NAC Score', fontsize=11)
            ax.set_title(f'{layer}', fontsize=12, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)
        
        plt.suptitle('NAC Score Comparison Across Experiments', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_delta_heatmap(self,
                          results: Dict[str, List[AnalysisResult]],
                          baseline_name: str = 'clean',
                          save_path: Optional[str] = None):
        """
        绘制相对于基线的Delta热力图
        
        Args:
            results: {experiment_name: [AnalysisResult]}
            baseline_name: 基线实验名称
            save_path: 保存路径
        """
        layers = list(set(r.layer_name for exp_results in results.values() for r in exp_results))
        exp_names = [name for name in results.keys() if name != baseline_name]
        
        # 获取基线
        baseline = {r.layer_name: r.mean for r in results[baseline_name]}
        
        # 计算Delta矩阵
        delta_matrix = np.zeros((len(exp_names), len(layers)))
        for i, exp_name in enumerate(exp_names):
            for j, layer in enumerate(layers):
                exp_result = next((r for r in results[exp_name] if r.layer_name == layer), None)
                if exp_result:
                    delta_matrix[i, j] = exp_result.mean - baseline[layer]
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(max(8, len(layers)*1.5), max(6, len(exp_names)*0.8)))
        
        im = ax.imshow(delta_matrix, cmap='RdYlGn_r', aspect='auto', vmin=-0.2, vmax=0.2)
        
        # 设置刻度
        ax.set_xticks(np.arange(len(layers)))
        ax.set_yticks(np.arange(len(exp_names)))
        ax.set_xticklabels(layers, fontsize=10)
        ax.set_yticklabels(exp_names, fontsize=10)
        
        # 旋转标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值标注
        for i in range(len(exp_names)):
            for j in range(len(layers)):
                text = ax.text(j, i, f'{delta_matrix[i, j]:+.3f}',
                             ha="center", va="center", color="white" if abs(delta_matrix[i, j]) > 0.1 else "black",
                             fontsize=9)
        
        ax.set_title(f'NAC Delta Heatmap (vs {baseline_name})', fontsize=14, fontweight='bold', pad=20)
        fig.colorbar(im, ax=ax, label='Δ NAC Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def generate_full_report(self,
                            results: Dict[str, List[AnalysisResult]],
                            output_dir: str = 'visualization_output',
                            baseline_name: str = 'clean'):
        """
        生成完整的可视化报告
        
        包括:
        - 每层的PDF直方图
        - 对比柱状图
        - Delta热力图
        - 统计摘要
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Generating Visualization Report to: {output_dir}")
        print('='*60)
        
        # 1. 每层的PDF直方图
        layers = list(set(r.layer_name for exp_results in results.values() for r in exp_results))
        for layer in layers:
            fig = self.plot_coverage_histogram(
                results, layer,
                save_path=str(output_path / f'histogram_{layer}.png')
            )
            plt.close(fig)
        
        # 2. 对比柱状图
        fig = self.plot_comparison_bars(
            results,
            baseline_name=baseline_name,
            save_path=str(output_path / 'comparison_bars.png')
        )
        plt.close(fig)
        
        # 3. Delta热力图
        fig = self.plot_delta_heatmap(
            results,
            baseline_name=baseline_name,
            save_path=str(output_path / 'delta_heatmap.png')
        )
        plt.close(fig)
        
        # 4. 统计摘要
        summary = self._generate_summary(results, baseline_name)
        with open(output_path / 'statistical_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved: {output_path / 'statistical_summary.json'}")
        
        print(f"{'='*60}")
        print(f"Report Generation Complete!")
        print(f"{'='*60}\n")
    
    def _generate_summary(self, results: Dict[str, List[AnalysisResult]], baseline_name: str) -> dict:
        """生成统计摘要"""
        summary = {
            'baseline': baseline_name,
            'experiments': {}
        }
        
        # 获取基线
        baseline = {r.layer_name: r.mean for r in results[baseline_name]}
        
        for exp_name, exp_results in results.items():
            exp_summary = {}
            for result in exp_results:
                layer_summary = {
                    'mean': result.mean,
                    'std': result.std,
                    'n_samples': len(result.scores),
                }
                if exp_name != baseline_name:
                    layer_summary['delta_vs_baseline'] = result.mean - baseline[result.layer_name]
                
                exp_summary[result.layer_name] = layer_summary
            
            summary['experiments'][exp_name] = exp_summary
        
        return summary
