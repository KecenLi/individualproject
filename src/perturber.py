import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import math
from autoattack import AutoAttack

# Import officially supported attacks from advex_uar
try:
    from advex_uar.attacks import ElasticAttack, FogAttack, SnowAttack, GaborAttack, JPEGAttack
    from advex_uar.attacks.attacks import IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.getcwd(), 'advex-uar'))
    from advex_uar.attacks import ElasticAttack, FogAttack, SnowAttack, GaborAttack, JPEGAttack
    from advex_uar.attacks.attacks import IMAGENET_MEAN, IMAGENET_STD

# ==========================================
# VERSION COMPATIBILITY LAYER
# ==========================================
if not hasattr(torch.Tensor, 'ndims'):
    torch.Tensor.ndims = property(lambda self: self.ndim)


def _imagenet_mean_std(device: torch.device, dtype: torch.dtype):
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def _normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean, std = _imagenet_mean_std(x.device, x.dtype)
    return (x - mean) / std


def _unnormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean, std = _imagenet_mean_std(x.device, x.dtype)
    return x * std + mean


class AdvexModelWrapper(nn.Module):
    """
    Advex-UAR attacks expect ImageNet-normalized inputs.
    RobustBench CIFAR models usually expect raw [0, 1] tensors.
    This wrapper cancels the normalization before calling the base model.
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x_norm.device, dtype=x_norm.dtype)
        std = self.std.to(device=x_norm.device, dtype=x_norm.dtype)
        x_plain = torch.clamp(x_norm * std + mean, 0.0, 1.0)
        return self.base_model(x_plain)


def _sample_uniform(lo: float, hi: float) -> float:
    """Sample a scalar uniformly from [lo, hi]."""
    return float(torch.empty(1).uniform_(float(lo), float(hi)).item())


def _resolve_range(value=None, value_range=None, default=0.0):
    """
    Resolve a scalar parameter with optional range support.
    Priority: explicit value -> range -> default.
    """
    if value is not None:
        return float(value)
    if value_range is not None and len(value_range) == 2:
        return _sample_uniform(value_range[0], value_range[1])
    return float(default)


class Perturber:
    """
    Perturbation Manager using pure Library APIs where available.
    """
    def __init__(self, model=None, device='cuda'):
        # Model is optional for pure, label-free transforms (e.g., noise/rotate).
        self.model = None
        self.advex_model = None
        if model is not None:
            # Ensure model is a proper nn.Module for advex_uar compatibility
            if not isinstance(model, nn.Module):
                class ModelWrapper(nn.Module):
                    def __init__(self, m):
                        super().__init__()
                        self.m = m
                        self.training = False
                    def forward(self, x): return self.m(x)
                self.model = ModelWrapper(model).to(device)
            else:
                self.model = model.to(device)
                if not hasattr(self.model, 'training'):
                    self.model.training = False

            self.model.eval()
            self.advex_model = AdvexModelWrapper(self.model).to(device)
            self.advex_model.eval()

        self.device = device
        self.configs = []

    def clear(self): self.configs = []

    def add(self, name, **params):
        self.configs.append((name, params))

    def apply(self, x, labels=None):
        x_adv = x.clone().detach().to(self.device).float()

        label_required = {'autoattack', 'elastic', 'fog', 'snow', 'gabor', 'jpeg'}

        # Prefer labels provided via params (e.g., autoattack config) if available.
        if labels is None:
            for name, params in self.configs:
                if 'labels' in params and params['labels'] is not None:
                    labels = params['labels']
                    break

        needs_labels = any(name in label_required for name, _ in self.configs)

        if labels is not None:
            labels = labels.to(self.device)
        elif needs_labels:
            if self.model is None:
                raise ValueError("Model is required for label-based attacks but model=None was provided.")
            with torch.no_grad():
                labels = self.model(x_adv).argmax(1)

        for name, params in self.configs:
            if name == 'autoattack':
                if self.model is None:
                    raise ValueError("AutoAttack requires a model, but model=None was provided.")
                adversary = AutoAttack(
                    self.model,
                    norm=params['norm'],
                    eps=params['eps'],
                    device=self.device,
                    version=params.get('version', 'standard'),
                    verbose=params.get('verbose', False),
                )
                if params.get('version') == 'fast':
                    adversary.attacks_to_run = ['apgd-ce']
                x_adv = adversary.run_standard_evaluation(x_adv, labels, bs=x_adv.shape[0])
            
            elif name == 'gaussian_noise':
                # Official PyTorch Noise Injection
                severity = params.get('severity', 0.1)
                noise = torch.randn_like(x_adv) * severity
                x_adv = torch.clamp(x_adv + noise, 0, 1)
                
            elif name == 'rotate':
                # Official Torchvision Rotation
                angle = _resolve_range(
                    value=params.get('angle', None),
                    value_range=params.get('angle_range', None),
                    default=0.0,
                )
                x_adv = torch.clamp(
                    F.affine(x_adv, angle=angle, translate=[0, 0], scale=1.0, shear=[0.0, 0.0]),
                    0,
                    1,
                )
            elif name in ['translate', 'translation']:
                dx = params.get('dx', None)
                dy = params.get('dy', None)
                max_dx = params.get('max_dx', None)
                max_dy = params.get('max_dy', None)
                dx_range = params.get('dx_range', None)
                dy_range = params.get('dy_range', None)

                if dx is None and max_dx is not None:
                    dx_range = (-float(max_dx), float(max_dx))
                if dy is None and max_dy is not None:
                    dy_range = (-float(max_dy), float(max_dy))

                dx = int(round(_resolve_range(dx, dx_range, default=0.0)))
                dy = int(round(_resolve_range(dy, dy_range, default=0.0)))
                x_adv = torch.clamp(
                    F.affine(x_adv, angle=0.0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0]),
                    0,
                    1,
                )
            elif name == 'scale':
                scale = params.get('scale', None)
                scale_range = params.get('scale_range', None)
                if scale_range is None and ('min_scale' in params and 'max_scale' in params):
                    scale_range = (params['min_scale'], params['max_scale'])
                scale = _resolve_range(scale, scale_range, default=1.0)
                x_adv = torch.clamp(
                    F.affine(x_adv, angle=0.0, translate=[0, 0], scale=scale, shear=[0.0, 0.0]),
                    0,
                    1,
                )
            elif name == 'shear':
                shear_x = params.get('shear_x', None)
                shear_y = params.get('shear_y', None)
                shear_x_range = params.get('shear_x_range', None)
                shear_y_range = params.get('shear_y_range', None)

                if 'shear' in params and shear_x is None and shear_y is None:
                    shear_val = params['shear']
                    if isinstance(shear_val, (list, tuple)) and len(shear_val) == 2:
                        shear_x, shear_y = shear_val
                    else:
                        shear_x = shear_val
                        shear_y = 0.0

                shear_x = _resolve_range(shear_x, shear_x_range, default=0.0)
                shear_y = _resolve_range(shear_y, shear_y_range, default=0.0)
                x_adv = torch.clamp(
                    F.affine(x_adv, angle=0.0, translate=[0, 0], scale=1.0, shear=[shear_x, shear_y]),
                    0,
                    1,
                )
            elif name == 'brightness':
                factor = params.get('factor', 1.0)
                x_adv = torch.clamp(F.adjust_brightness(x_adv, factor), 0, 1)
            elif name == 'contrast':
                factor = params.get('factor', 1.0)
                x_adv = torch.clamp(F.adjust_contrast(x_adv, factor), 0, 1)

            elif name in ['elastic']:
                if self.model is None:
                    raise ValueError("Elastic attack requires a model, but model=None was provided.")
                # Advex-UAR ElasticAttack requires strict params mapping
                # Init signature: (nb_its, eps_max, step_size, resol, ...)
                # Call signature: (model, img, target)
                
                # CIFAR-10 resolution
                resol = 32
                nb_its = params.get('n_iters', 10)
                eps_max = params.get('eps', 16.0) # Pixel space [0, 255]? No, advex-uar uses pixels for eps_max
                step_size = params.get('step_size', eps_max / math.sqrt(nb_its)) # Heuristic from paper

                # Advex-UAR expects normalized inputs and returns normalized outputs.
                x_norm = _normalize_imagenet(x_adv)
                attacker = ElasticAttack(nb_its=nb_its, eps_max=eps_max, step_size=step_size, resol=resol)

                x_adv_norm = attacker(self.advex_model, x_norm, labels)
                x_adv = torch.clamp(_unnormalize_imagenet(x_adv_norm), 0.0, 1.0)

            elif name in ['fog']:
                if self.model is None:
                    raise ValueError("Fog attack requires a model, but model=None was provided.")
                # FogAttack(nb_its, eps_max, step_size, resol, ...)
                resol = 32
                nb_its = params.get('n_iters', 10)
                eps_max = params.get('eps', 255.0) # Fog usually full range
                step_size = params.get('step_size', 2.5) 

                x_norm = _normalize_imagenet(x_adv)
                attacker = FogAttack(nb_its=nb_its, eps_max=eps_max, step_size=step_size, resol=resol)
                x_adv_norm = attacker(self.advex_model, x_norm, labels)
                x_adv = torch.clamp(_unnormalize_imagenet(x_adv_norm), 0.0, 1.0)
                
            elif name in ['snow']:
                if self.model is None:
                    raise ValueError("Snow attack requires a model, but model=None was provided.")
                # SnowAttack(nb_its, eps_max, step_size, resol, ...)
                resol = 32
                nb_its = params.get('n_iters', 10)
                eps_max = params.get('eps', 0.1) # Snow eps is different scale usually?
                step_size = params.get('step_size', eps_max / math.sqrt(nb_its))

                x_norm = _normalize_imagenet(x_adv)
                attacker = SnowAttack(nb_its=nb_its, eps_max=eps_max, step_size=step_size, resol=resol)
                x_adv_norm = attacker(self.advex_model, x_norm, labels)
                x_adv = torch.clamp(_unnormalize_imagenet(x_adv_norm), 0.0, 1.0)

            elif name == 'gabor':
                if self.model is None:
                    raise ValueError("Gabor attack requires a model, but model=None was provided.")
                # GaborAttack(nb_its, eps_max, step_size, resol, rand_init=True, scale_each=False)
                resol = 32
                nb_its = params.get('n_iters', 10)
                eps_max = params.get('eps', 40.0) # Pixel space
                step_size = params.get('step_size', eps_max / math.sqrt(nb_its))

                x_norm = _normalize_imagenet(x_adv)
                attacker = GaborAttack(nb_its=nb_its, eps_max=eps_max, step_size=step_size, resol=resol)
                x_adv_norm = attacker(self.advex_model, x_norm, labels)
                x_adv = torch.clamp(_unnormalize_imagenet(x_adv_norm), 0.0, 1.0)

            elif name == 'jpeg':
                if self.model is None:
                    raise ValueError("JPEG attack requires a model, but model=None was provided.")
                # JPEGAttack(nb_its, eps_max, step_size, resol, rand_init=True, opt='linf', ...)
                # opt can be 'linf', 'l2', or 'l1'
                resol = 32
                nb_its = params.get('n_iters', 10)
                eps_max = params.get('eps', 0.125) # Default from paper
                step_size = params.get('step_size', eps_max / math.sqrt(nb_its))
                opt = params.get('norm', 'linf').lower() 

                x_norm = _normalize_imagenet(x_adv)
                attacker = JPEGAttack(nb_its=nb_its, eps_max=eps_max, step_size=step_size, resol=resol, opt=opt)
                # JPEGAttack default is avoid_target=False (targeted), so we must explicitly set True
                x_adv_norm = attacker(self.advex_model, x_norm, labels, avoid_target=True)
                x_adv = torch.clamp(_unnormalize_imagenet(x_adv_norm), 0.0, 1.0)

        return x_adv


def apply_ordered_perturbations(images, transforms, model=None, device='cuda', labels=None):
    """
    Compose perturbations in the provided order using the existing Perturber.
    This is a thin orchestration layer that keeps the implementation library-first.
    """
    if not transforms:
        return images.to(device)

    perturber = Perturber(model=model, device=device)
    for name, params in transforms:
        perturber.add(name, **params)

    return perturber.apply(images, labels=labels)
