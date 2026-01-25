"""
高效NAC分析器 - 基于原始ood_coverage库的正确实现

核心优化：
1. 使用原生32x32 CIFAR-10尺寸（而非resize到224x224）
2. 采用两阶段策略：Profiling + Testing
3. 正确的梯度计算和内存管理
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class AnalysisResult:
    """分析结果容器"""
    perturbation_name: str
    layer_name: str
    scores: List[float]
    mean: float
    std: float


from .nethook import InstrumentedModel


def kl_grad(b_state, outputs, retain_graph=False):
    """
    计算KL散度对layer状态的梯度
    来自原始库: openood/postprocessors/nac/instr_state.py
    """
    logsoftmax = nn.LogSoftmax(dim=-1)
    num_classes = outputs.shape[-1]
    targets = torch.ones_like(outputs) / num_classes
    
    loss = torch.sum(-targets * logsoftmax(outputs), dim=-1)
    layer_grad = torch.autograd.grad(loss.sum(), b_state, 
                                     create_graph=False,
                                     retain_graph=retain_graph)[0]
    return layer_grad


def avg_pooling(hiddens, method="avg"):
    """空间平均池化 - 将 (B, C, H, W) 转为 (B, C)"""
    if len(hiddens.shape) == 4:
        if method == "avg":
            return hiddens.mean(dim=[2, 3])
        else:
            return hiddens.sum(dim=[2, 3])
    return hiddens


def logspace_thresholds(base=1e3, num=100, device='cuda'):
    """生成对数空间阈值 - 来自原始库"""
    num = int(num / 2)
    x = np.linspace(1, np.sqrt(base), num=num)
    x_l = np.emath.logn(base, x)
    x_r = (1 - x_l)[::-1]
    x = np.concatenate([x_l[:-1], x_r])
    x[-1] += 1e-2
    thresholds = np.append(x, 1.2)
    return torch.from_numpy(thresholds).float().to(device)


@dataclass
class LayerProfile:
    """层的激活分布Profile"""
    thresh: torch.Tensor  # (M,) 阈值
    t_act: torch.Tensor   # (M-1, N) 每个区间的激活次数
    n_coverage: Optional[torch.Tensor] = None  # 归一化后的覆盖率


class EfficientNACAnalyzer:
    """
    高效NAC分析器
    
    使用两阶段策略：
    1. Profiling: 在训练集子集上建立每个神经元的激活分布
    2. Testing: 用已建立的分布快速评估测试样本
    """
    
    def __init__(self, model, layer_names: List[str], device='cuda',
                 M=50, O=50, sig_alpha=100.0):
        """
        Args:
            model: 模型
            layer_names: 要监控的层名称
            device: 设备
            M: 直方图区间数量 (对齐配置 M=50)
            O: 最小激活阈值 (针对1000个样本，对齐配置 O=50)
            sig_alpha: sigmoid的alpha参数 (对齐配置 sig_alpha=100)
        """
        self.model = model.to(device)
        self.model.eval()
        self.layer_names = layer_names
        self.device = device
        self.M = M
        self.O = O
        self.sig_alpha = sig_alpha
        self.profiles: Dict[str, LayerProfile] = {}
        self.is_profiled = False
    
    def _compute_nac_states(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算NAC状态 (z * grad) 用sigmoid激活
        
        Returns:
            {layer_name: states (B, N)}
        """
        x = x.to(self.device)
        states = {}
        
        with InstrumentedModel(self.model) as instr:
            instr.retain_layers(self.layer_names, detach=False)
            out = self.model(x)
            
            for i, ln in enumerate(self.layer_names):
                retain_graph = (i < len(self.layer_names) - 1)
                b_state = instr.retained_layer(ln)
                b_grad = kl_grad(b_state, out, retain_graph=retain_graph)
                
                # 池化
                b_state_p = avg_pooling(b_state).detach()
                b_grad_p = avg_pooling(b_grad).detach()
                
                # NAC公式: sigmoid(z * grad)
                nac_state = torch.sigmoid(self.sig_alpha * b_state_p * b_grad_p)
                states[ln] = nac_state
        
        return states
    
    def profile(self, dataloader, max_samples: int = 1000, desc="Profiling"):
        """
        阶段1: 在数据集上建立激活分布Profile
        
        Args:
            dataloader: 数据加载器
            max_samples: 最大样本数
            desc: 进度条描述
        """
        print(f"[NAC Profiling] 使用 {max_samples} 个样本建立分布...")
        
        # 初始化
        layer_sizes = {}
        layer_activations = {ln: [] for ln in self.layer_names}
        n_samples = 0
        
        for images, labels in tqdm(dataloader, desc=desc):
            if n_samples >= max_samples:
                break
            
            states = self._compute_nac_states(images)
            
            for ln in self.layer_names:
                layer_activations[ln].append(states[ln].cpu())
                if ln not in layer_sizes:
                    layer_sizes[ln] = states[ln].shape[1]
            
            n_samples += images.shape[0]
            torch.cuda.empty_cache()
        
        # 构建Profile
        thresh = logspace_thresholds(base=1e3, num=self.M, device=self.device)
        
        for ln in self.layer_names:
            all_states = torch.cat(layer_activations[ln], dim=0).to(self.device)  # (N_samples, N_neurons)
            N = all_states.shape[1]
            
            # 统计每个区间的激活次数
            t_act = torch.zeros(self.M - 1, N, device=self.device)
            
            for i in range(self.M - 1):
                mask = (all_states >= thresh[i]) & (all_states < thresh[i + 1])
                t_act[i] = mask.sum(dim=0).float()
            
            # 归一化
            t_score = torch.clamp(t_act / self.O, max=1.0)
            
            self.profiles[ln] = LayerProfile(
                thresh=thresh,
                t_act=t_act,
                n_coverage=t_score
            )
            
            coverage = t_score.mean()
            print(f"  {ln}: {N} neurons, coverage={coverage:.4f}")
        
        self.is_profiled = True
        print("[NAC Profiling] 完成!")
    
    def score_batch(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        阶段2: 使用已建立的Profile对批次打分
        
        与原始库对齐：ood_test() 方法
        
        Args:
            images: 输入图像 (B, C, H, W)
            
        Returns:
            {layer_name: scores (B,)} 分数在 [0, 1] 范围
        """
        if not self.is_profiled:
            raise RuntimeError("必须先调用 profile() 建立分布！")
        
        states = self._compute_nac_states(images)
        scores = {}
        
        for ln in self.layer_names:
            profile = self.profiles[ln]
            batch_states = states[ln]  # (B, N)
            B, N = batch_states.shape
            
            # 与原始库对齐：计算每个样本在每个神经元上的覆盖率得分
            # b_act: (B, M-1, N) - 判断每个样本在每个神经元上落入哪个区间
            # n_coverage: (M-1, N) - 每个区间每个神经元的归一化覆盖率
            
            # 初始化 scores: (B, N)
            sample_neuron_scores = torch.zeros(B, N, device=self.device)
            
            for i in range(self.M - 1):
                # 判断激活落在哪个区间: (B, N)
                mask = (batch_states >= profile.thresh[i]) & \
                       (batch_states < profile.thresh[i + 1])
                # 加权: (B, N) * (N,) -> (B, N)
                sample_neuron_scores += mask.float() * profile.n_coverage[i]
            
            # 跨神经元取平均: (B, N) -> (B,)
            # 这与原始库的 scores.mean(dim=1) 对应
            batch_scores = sample_neuron_scores.mean(dim=1)
            
            scores[ln] = batch_scores
        
        return scores
    
    def analyze_dataset(self, dataloader, max_samples: int = None, 
                       desc="Analyzing") -> Dict[str, np.ndarray]:
        """
        分析整个数据集
        
        Returns:
            {layer_name: scores array}
        """
        all_scores = {ln: [] for ln in self.layer_names}
        n_samples = 0
        
        for images, labels in tqdm(dataloader, desc=desc):
            if max_samples and n_samples >= max_samples:
                break
            
            scores = self.score_batch(images)
            
            for ln in self.layer_names:
                all_scores[ln].extend(scores[ln].cpu().numpy().tolist())
            
            n_samples += images.shape[0]
            torch.cuda.empty_cache()
        
        return {ln: np.array(scores) for ln, scores in all_scores.items()}


def quick_nac_test(model, dataloader, layer_name: str, 
                   profile_samples: int = 500,
                   test_samples: int = 500,
                   device='cuda') -> Dict:
    """
    快速NAC测试（一站式接口）
    
    Returns:
        {'mean': float, 'std': float, 'scores': np.ndarray}
    """
    analyzer = EfficientNACAnalyzer(model, [layer_name], device)
    
    # Profiling
    analyzer.profile(dataloader, max_samples=profile_samples)
    
    # Testing
    scores = analyzer.analyze_dataset(dataloader, max_samples=test_samples)
    
    layer_scores = scores[layer_name]
    return {
        'mean': float(layer_scores.mean()),
        'std': float(layer_scores.std()),
        'scores': layer_scores
    }
