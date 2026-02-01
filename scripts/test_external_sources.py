"""
System-level smoke tests for DeepG and OODRobustBench integrations.

By default this script avoids heavy downloads and skips tests if datasets
or libraries are missing. Use --allow-download to enable dataset downloads.
"""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch

from src.external_sources import DeepGAttackGenerator, OODRobustBenchSource


def test_deepg(config_path: str, deepg_root: str | None, n_examples: int, attack_index: int) -> bool:
    print("[DeepG] checking availability...")
    generator = DeepGAttackGenerator(
        config_path=config_path,
        deepg_root=deepg_root,
        attack_index=attack_index,
    )
    if not generator.is_available():
        print("[DeepG] SKIP: libgeometric.so not found. Build with (cd libs/deepg && make shared_object).")
        return False

    print("[DeepG] generating samples...")
    loader = generator.as_dataloader(range(n_examples), batch_size=min(4, n_examples))
    batch = next(iter(loader))
    x, y = batch
    print(f"[DeepG] OK: batch shape={tuple(x.shape)} labels={y[:5].tolist()}")
    return True


def test_oodrb_corruption(data_dir: str, allow_download: bool, n_examples: int) -> bool:
    print("[OODRobustBench] checking corruption loader...")
    data_root = Path(data_dir) / "CIFAR-10-C"
    if not data_root.exists() and not allow_download:
        print("[OODRobustBench] SKIP: CIFAR-10-C not found. Re-run with --allow-download to fetch.")
        return False

    source = OODRobustBenchSource(data_dir=data_dir)
    x, y = source.load_corruption(
        dataset="cifar10",
        corruption="gaussian_noise",
        severity=1,
        n_examples=n_examples,
        model_name="Standard",
        threat_model="Linf",
    )
    print(f"[OODRobustBench] OK: corruption batch shape={tuple(x.shape)} labels={y[:5].tolist()}")
    return True


def test_oodrb_natural_shift(data_dir: str, allow_download: bool, n_examples: int) -> bool:
    print("[OODRobustBench] checking natural shift loader...")
    # CIFAR-10.1 is stored under data/cifar-10.1 by default.
    data_root = Path(data_dir) / "cifar-10.1"
    if not data_root.exists() and not allow_download:
        print("[OODRobustBench] SKIP: CIFAR-10.1 not found. Re-run with --allow-download and prepare data.")
        return False

    source = OODRobustBenchSource(data_dir=data_dir)
    x, y = source.load_natural_shift(
        dataset="cifar10",
        shift="cifar10.1",
        n_examples=n_examples,
        model_name="Standard",
        threat_model="Linf",
    )
    print(f"[OODRobustBench] OK: natural shift batch shape={tuple(x.shape)} labels={y[:5].tolist()}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--deepg-config", type=str, default="configs/deepg_cifar10_rotate_small.txt")
    parser.add_argument("--deepg-root", type=str, default=None)
    parser.add_argument("--attack-index", type=int, default=0)
    parser.add_argument("--n-examples", type=int, default=4)
    args = parser.parse_args()

    print("== External Sources System Test ==")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")

    deepg_ok = test_deepg(args.deepg_config, args.deepg_root, args.n_examples, args.attack_index)
    oodrb_corr_ok = test_oodrb_corruption(args.data_dir, args.allow_download, args.n_examples)
    oodrb_ns_ok = test_oodrb_natural_shift(args.data_dir, args.allow_download, args.n_examples)

    print("== Summary ==")
    print(f"DeepG: {'OK' if deepg_ok else 'SKIP/FAIL'}")
    print(f"OODRB Corruption: {'OK' if oodrb_corr_ok else 'SKIP/FAIL'}")
    print(f"OODRB Natural Shift: {'OK' if oodrb_ns_ok else 'SKIP/FAIL'}")


if __name__ == "__main__":
    main()
