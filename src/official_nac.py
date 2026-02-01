import sys
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any
from unittest.mock import MagicMock
from tqdm import tqdm
import yaml

# Mock optional OpenOOD dependencies for local imports.
missing_deps = [
    'faiss', 'faiss.contrib', 'faiss.contrib.torch_utils',
    'diffdist', 'diffdist.functional',
    'libmr', 'cvxpy', 'cvxopt', 'sklearn.covariance.EmpiricalCovariance',
    'cv2', 'skimage', 'imgaug', 'imgaug.augmenters'
]
for dep in missing_deps:
    sys.modules[dep] = MagicMock()

# Ensure ood_coverage is on sys.path.
sys.path.append(os.path.join(os.getcwd(), 'ood_coverage'))

# Import the module with get_intr_name.
import openood.postprocessors.nac.instr_state as instr_state

# Define and apply the monkeypatch.
def patched_get_intr_name(layer_names, model_name, network=None):
    from collections import OrderedDict
    aka_name_dict = OrderedDict()
    for ln in layer_names:
        aka_name_dict[ln] = ln
    print(f"[MonkeyPatch] Bypassed get_intr_name logic. Using layers: {layer_names}")
    return aka_name_dict, list(aka_name_dict.values())

instr_state.get_intr_name = patched_get_intr_name

# Import NACPostprocessor after patching.
from openood.postprocessors import NACPostprocessor
from openood.evaluators.metrics import compute_all_metrics

# Patch any cached references inside the module.
import openood.postprocessors.nac_postprocessor as npm
npm.get_intr_name = patched_get_intr_name

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict): value = DotDict(value)
            self[key] = value

class DictDatasetWrapper(Dataset):
    """Adapt datasets to OpenOOD {data, label} format."""
    def __init__(self, dataset):
        self.dataset = dataset
        if hasattr(dataset, 'targets'): self.targets = dataset.targets
        elif hasattr(dataset, 'labels'): self.targets = dataset.labels
        else: self.targets = [0] * len(dataset)
        self.samples = [(None, t) for t in self.targets]
        
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index):
        data, label = self.dataset[index]
        return {'data': data, 'label': label}

class OfficialNACWrapper:
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model
        self.device = device
        
        # Load official config.
        config_path = 'ood_coverage/configs/postprocessors/nac/resnet/nac_cifar10.yml'
        with open(config_path, 'r') as f:
            raw_cfg = yaml.safe_load(f)
        
        if 'network' not in raw_cfg: raw_cfg['network'] = {'name': 'resnet18'}
        self.config = DotDict(raw_cfg)
        
        # Initialize official API.
        self.postprocessor = NACPostprocessor(self.config)

    def setup(self, train_loader: DataLoader, layer_names: List[str], valid_num: int = 1000):
        """Mirror Postprocessor.setup call path."""
        
        # Fill missing layer configs.
        default_kwargs = self.config.postprocessor.layer_kwargs.get('avgpool', {})
        default_sweep = self.config.postprocessor.postprocessor_sweep.get('avgpool', {})
        
        for ln in layer_names:
            if ln not in self.config.postprocessor.layer_kwargs:
                self.config.postprocessor.layer_kwargs[ln] = dict(default_kwargs)
                self.config.postprocessor.postprocessor_sweep[ln] = dict(default_sweep)
        
        wrapped_train_loader = DataLoader(
            DictDatasetWrapper(train_loader.dataset), 
            batch_size=train_loader.batch_size, num_workers=0
        )
        
        # Call the official setup API.
        self.postprocessor.setup(
            self.model, 
            id_loader_dict={'main_train': wrapped_train_loader},
            ood_loader_dict=None,
            id_name='cifar10',
            valid_num=valid_num,
            layer_names=layer_names,
            aps=False 
        )

    def run_aps(self, id_val_loader: DataLoader, ood_val_loader: DataLoader):
        """Run APS using the official hyperparam search flow."""
        print("\n[Official API] Starting Parameter Search...")
        
        id_val = DataLoader(DictDatasetWrapper(id_val_loader.dataset), batch_size=32)
        ood_val = DataLoader(DictDatasetWrapper(ood_val_loader.dataset), batch_size=32)
        
        p = self.postprocessor
        # Use official args_dict structure.
        hyperparam_names = list(p.args_dict.keys())
        hyperparam_list = [p.args_dict[name] for name in hyperparam_names]
        
        # Generate combinations.
        def recursive_generator(lp, n):
            if n == 1: return [[x] for x in lp[0]]
            res = []
            for x in lp[n-1]:
                for y in recursive_generator(lp, n-1):
                    k = y.copy(); k.append(x); res.append(k)
            return res

        combinations = recursive_generator(hyperparam_list, len(hyperparam_names))
        print(f"Sweeping over {len(combinations)} combinations using Official compute_all_metrics API...")
        
        max_auroc = -1
        best_config = None
        
        for hp in tqdm(combinations, desc="APS Sweep"):
            p.set_hyperparam(hp)
            p.build_nac(self.model)
            
            # Official inference API.
            id_preds, id_confs, _ = p.inference(self.model, id_val, progress=False)
            ood_preds, ood_confs, _ = p.inference(self.model, ood_val, progress=False)
            
            # Compute metrics with official labels.
            y_scores = np.concatenate([id_confs, ood_confs])
            y_labels = np.concatenate([np.ones_like(id_confs), -1 * np.ones_like(ood_confs)])
            y_preds = np.concatenate([id_preds, ood_preds])
            
            metrics = compute_all_metrics(y_scores, y_labels, y_preds)
            auroc = metrics[1]
            
            if auroc > max_auroc:
                max_auroc = auroc
                best_config = hp
        
        print(f"APS Done. Best AUROC recorded: {max_auroc:.4f}")
        p.set_hyperparam(best_config)
        p.build_nac(self.model)

    def score_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Get confidence scores via the official API."""
        class SimpleBatchDataset(Dataset):
            def __init__(self, x): self.x = x
            def __len__(self): return len(self.x)
            def __getitem__(self, i): return {'data': self.x[i], 'label': torch.tensor(0)}
            
        temp_loader = DataLoader(SimpleBatchDataset(images), batch_size=len(images))
        # inference returns (preds, confs, labels)
        _, scores, _ = self.postprocessor.inference(self.model, temp_loader, progress=False)
        
        if isinstance(scores, torch.Tensor): return scores.to(self.device).float()
        return torch.from_numpy(scores).to(self.device).float()
