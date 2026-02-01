"""
Efficient NAC analyzer aligned with the official ood_coverage logic.
Uses a profiling/testing split and native 32x32 CIFAR-10 inputs.
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class AnalysisResult:
    """Container for analysis outputs."""
    perturbation_name: str
    layer_name: str
    scores: List[float]
    mean: float
    std: float


from .nethook import InstrumentedModel


def kl_grad(b_state, outputs, retain_graph=False):
    """
    KL-divergence gradient w.r.t. layer state.
    Mirrors openood/postprocessors/nac/instr_state.py.
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
    """Spatial pooling: (B, C, H, W) -> (B, C)."""
    if len(hiddens.shape) == 4:
        if method == "avg":
            return hiddens.mean(dim=[2, 3])
        else:
            return hiddens.sum(dim=[2, 3])
    return hiddens


def logspace_thresholds(base=1e3, num=100, device='cuda'):
    """Log-space thresholds from the official implementation."""
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
    """Layer activation profile."""
    thresh: torch.Tensor
    t_act: torch.Tensor
    n_coverage: Optional[torch.Tensor] = None


class EfficientNACAnalyzer:
    """
    Efficient NAC analyzer.
    Profiling builds activation bins; testing scores new samples.
    """
    
    def __init__(self, model, layer_names: List[str], device='cuda',
                 M=50, O=50, sig_alpha=100.0):
        """
        Args:
            model: torch.nn.Module
            layer_names: layers to monitor
            device: device string
            M: histogram bins
            O: minimum activation threshold
            sig_alpha: sigmoid scale
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
        Compute NAC states: sigmoid(alpha * z * grad).

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
                
                b_state_p = avg_pooling(b_state).detach()
                b_grad_p = avg_pooling(b_grad).detach()

                nac_state = torch.sigmoid(self.sig_alpha * b_state_p * b_grad_p)
                states[ln] = nac_state
        
        return states
    
    def profile(self, dataloader, max_samples: int = 1000, desc="Profiling"):
        """
        Stage 1: build activation profiles from a dataset subset.
        """
        print(f"[NAC Profiling] 使用 {max_samples} 个样本建立分布...")
        
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
        
        thresh = logspace_thresholds(base=1e3, num=self.M, device=self.device)
        
        for ln in self.layer_names:
            all_states = torch.cat(layer_activations[ln], dim=0).to(self.device)  # (N_samples, N_neurons)
            N = all_states.shape[1]
            
            t_act = torch.zeros(self.M - 1, N, device=self.device)
            
            for i in range(self.M - 1):
                mask = (all_states >= thresh[i]) & (all_states < thresh[i + 1])
                t_act[i] = mask.sum(dim=0).float()
            
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
        Stage 2: score a batch with the learned profiles.

        Args:
            images: (B, C, H, W)
        Returns:
            {layer_name: scores (B,)} in [0, 1]
        """
        if not self.is_profiled:
            raise RuntimeError("Call profile() before score_batch().")
        
        states = self._compute_nac_states(images)
        scores = {}
        
        for ln in self.layer_names:
            profile = self.profiles[ln]
            batch_states = states[ln]  # (B, N)
            B, N = batch_states.shape
            
            sample_neuron_scores = torch.zeros(B, N, device=self.device)
            
            for i in range(self.M - 1):
                mask = (batch_states >= profile.thresh[i]) & \
                       (batch_states < profile.thresh[i + 1])
                sample_neuron_scores += mask.float() * profile.n_coverage[i]
            
            batch_scores = sample_neuron_scores.mean(dim=1)
            
            scores[ln] = batch_scores
        
        return scores
    
    def analyze_dataset(self, dataloader, max_samples: int = None, 
                       desc="Analyzing") -> Dict[str, np.ndarray]:
        """
        Score an entire dataset.

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
    Quick NAC test helper.
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
