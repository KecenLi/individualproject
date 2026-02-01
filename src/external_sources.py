import os
import sys
import types
import importlib.util
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DeepGConfig:
    dataset: str
    dataset_split: str
    num_attacks: int


def _parse_deepg_config(config_path: str) -> DeepGConfig:
    dataset = None
    dataset_split = None
    num_attacks = None
    with open(config_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0]
            value = " ".join(parts[1:])
            if key == "dataset":
                dataset = value
            elif key == "set":
                dataset_split = value
            elif key == "num_attacks":
                try:
                    num_attacks = int(value)
                except ValueError:
                    num_attacks = None
    if dataset is None:
        dataset = "cifar10"
    if dataset_split is None:
        dataset_split = "test"
    if num_attacks is None:
        num_attacks = 1
    return DeepGConfig(dataset=dataset, dataset_split=dataset_split, num_attacks=num_attacks)


def _deepg_dataset_shape(dataset: str) -> Tuple[int, int, int]:
    if dataset in {"cifar10"}:
        return 32, 32, 3
    if dataset in {"mnist", "fashion"}:
        return 28, 28, 1
    if dataset in {"imagenet"}:
        return 250, 250, 3
    raise ValueError(f"Unsupported DeepG dataset: {dataset}")


def _deepg_dataset_csv_path(deepg_root: Path, dataset: str, split: str) -> Path:
    return deepg_root / "code" / "datasets" / f"{dataset}_{split}.csv"


def _read_deepg_labels(csv_path: Path, indices: Iterable[int]) -> List[int]:
    wanted = set(int(i) for i in indices)
    labels = {}
    if not wanted:
        return []
    max_idx = max(wanted)
    with open(csv_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > max_idx:
                break
            if i in wanted:
                label_str = line.split(",", 1)[0]
                labels[i] = int(label_str)
    return [labels[i] for i in indices]


@contextmanager
def _temp_cwd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class DeepGAttackGenerator:
    """
    Wrapper around the official DeepG geometric_constraints API.

    This generator uses DeepG to produce attack images (interval-valued outputs)
    and converts them into concrete tensors by taking the midpoint of [inf, sup].
    """

    def __init__(
        self,
        config_path: str,
        deepg_root: Optional[str] = None,
        attack_index: int = 0,
    ) -> None:
        self.config_path = str(config_path)
        self.attack_index = int(attack_index)
        self.deepg_root = Path(deepg_root) if deepg_root else Path(__file__).resolve().parents[1] / "libs" / "deepg"
        self._api = None
        self._container = None
        self._cfg = _parse_deepg_config(self.config_path)

    def is_available(self) -> bool:
        return (self.deepg_root / "build" / "libgeometric.so").exists()

    def _load_api(self):
        if self._api is not None:
            return self._api

        build_dir = self.deepg_root / "build"
        lib_path = build_dir / "libgeometric.so"
        if not lib_path.exists():
            raise FileNotFoundError(
                f"DeepG shared library not found at {lib_path}. "
                "Build it with: (cd libs/deepg && make shared_object) and ensure GUROBI is configured."
            )

        # Ensure DeepG Python API is importable
        if str(self.deepg_root) not in sys.path:
            sys.path.insert(0, str(self.deepg_root))

        # DeepG loads libgeometric.so by name; ensure the loader can find it
        gurobi_home = os.environ.get("GUROBI_HOME", "")
        gurobi_lib = f"{gurobi_home}/lib" if gurobi_home else ""
        lib_paths = [str(build_dir)]
        if gurobi_lib:
            lib_paths.append(gurobi_lib)
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_paths + ([existing] if existing else []))

        with _temp_cwd(self.deepg_root):
            import geometric_constraints as gc
        self._api = gc
        return gc

    def _get_container(self):
        if self._container is not None:
            return self._container
        api = self._load_api()
        with _temp_cwd(self.deepg_root):
            container = api.get_transform_attack_container(self.config_path)
        self._container = container
        return container

    def generate_for_indices(self, indices: Iterable[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = [int(i) for i in indices]
        if not indices:
            return torch.empty(0), torch.empty(0, dtype=torch.long)

        cfg = self._cfg
        n_rows, n_cols, n_channels = _deepg_dataset_shape(cfg.dataset)
        csv_path = _deepg_dataset_csv_path(self.deepg_root, cfg.dataset, cfg.dataset_split)
        labels = _read_deepg_labels(csv_path, indices)

        api = self._load_api()
        container = self._get_container()

        images: List[torch.Tensor] = []
        for idx in indices:
            with _temp_cwd(self.deepg_root):
                api.set_transform_attack_for(container, idx, attack=True, verbose=False)
                attack_images = api.get_attack_images(container)

            if not attack_images:
                raise RuntimeError("DeepG returned no attack images for index %s" % idx)

            chosen = attack_images[min(self.attack_index, len(attack_images) - 1)]
            vec = np.asarray(chosen, dtype=np.float32)
            if vec.size != n_rows * n_cols * n_channels * 2:
                raise ValueError(
                    f"DeepG attack image size mismatch: got {vec.size}, expected {n_rows * n_cols * n_channels * 2}"
                )
            arr = vec.reshape(n_rows, n_cols, n_channels, 2)
            img = (arr[..., 0] + arr[..., 1]) / 2.0
            img = np.clip(img, 0.0, 1.0)
            tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
            images.append(tensor)

        x = torch.stack(images, dim=0)
        y = torch.tensor(labels, dtype=torch.long)
        return x, y

    def as_dataloader(self, indices: Iterable[int], batch_size: int = 32) -> DataLoader:
        x, y = self.generate_for_indices(indices)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class OODRobustBenchSource:
    """
    Adapter for OODRobustBench shifts using the official dataset loaders.
    """

    def __init__(self, data_dir: str = "./data") -> None:
        self.data_dir = data_dir
        self._oodrb_data = self._load_oodrb_data_module()

    def _load_oodrb_data_module(self):
        """
        Load oodrobustbench.data without importing the full package __init__
        (which triggers heavy dependencies such as perceptual-advex).
        """
        oodrb_root = Path(__file__).resolve().parents[1] / "OODRobustBench" / "oodrobustbench"
        data_path = oodrb_root / "data.py"

        pkg_name = "oodrobustbench"
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(oodrb_root)]
            sys.modules[pkg_name] = pkg

        module_name = f"{pkg_name}.data"
        if module_name in sys.modules:
            return sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, str(data_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load oodrobustbench.data from {data_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def load_corruption(self,
                        dataset: str,
                        corruption: str,
                        severity: int,
                        n_examples: int,
                        model_name: str,
                        threat_model: str = "Linf") -> Tuple[torch.Tensor, torch.Tensor]:
        from robustbench.data import CORRUPTION_DATASET_LOADERS, get_preprocessing
        from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

        dataset_enum = BenchmarkDataset(dataset)
        threat_enum = ThreatModel(threat_model)
        prepr = get_preprocessing(dataset_enum, threat_enum, model_name, "Res224")
        loader_fn = CORRUPTION_DATASET_LOADERS[dataset_enum][ThreatModel.corruptions]
        x, y = loader_fn(n_examples, severity, self.data_dir, False, [corruption], prepr)
        return x, y

    def load_natural_shift(self,
                           dataset: str,
                           shift: str,
                           n_examples: int,
                           model_name: str,
                           threat_model: str = "Linf") -> Tuple[torch.Tensor, torch.Tensor]:
        from robustbench.data import get_preprocessing
        from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

        dataset_enum = BenchmarkDataset(dataset)
        threat_enum = ThreatModel(threat_model)
        prepr = get_preprocessing(dataset_enum, threat_enum, model_name, "Res224")
        x, y = self._oodrb_data.load_natural_shift_data(self.data_dir, dataset, shift, n_examples, prepr)
        return x, y

    def as_dataloader(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 64) -> DataLoader:
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
