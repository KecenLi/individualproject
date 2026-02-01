import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from robustbench.utils import load_model

from src.external_sources import DeepGAttackGenerator, OODRobustBenchSource

def get_resnet18(model_name='Standard'):
    """
    Load a CIFAR-10 model via RobustBench.
    """
    print(f"Loading {model_name} model via RobustBench API...")
    return load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')

def get_cifar10_loader(
    batch_size=128,
    train=False,
    n_examples=None,
    data_dir="./data",
    shuffle=None,
):
    """
    Use torchvision CIFAR-10 to separate train/test explicitly.
    RobustBench models typically include normalization, so keep [0, 1] tensors.
    """
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform,
    )

    if n_examples is not None:
        n_examples = min(int(n_examples), len(dataset))
        dataset = Subset(dataset, range(n_examples))

    if shuffle is None:
        shuffle = bool(train)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def get_deepg_loader(
    config_path: str,
    n_examples: int = 256,
    batch_size: int = 32,
    deepg_root: str | None = None,
    attack_index: int = 0,
):
    """
    Load DeepG-generated attack images using the official DeepG API.
    This returns a DataLoader of (images, labels) derived from DeepG's CSV dataset.
    """
    generator = DeepGAttackGenerator(
        config_path=config_path,
        deepg_root=deepg_root,
        attack_index=attack_index,
    )
    if not generator.is_available():
        raise FileNotFoundError(
            "DeepG libgeometric.so not found. Build it with: (cd libs/deepg && make shared_object)"
        )
    indices = list(range(int(n_examples)))
    return generator.as_dataloader(indices, batch_size=batch_size)


def get_oodrb_corruption_loader(
    dataset: str,
    corruption: str,
    severity: int,
    n_examples: int,
    model_name: str,
    threat_model: str = "Linf",
    batch_size: int = 64,
    data_dir: str = "./data",
):
    """
    Load OODRobustBench corruption data using the official OODRobustBench+RobustBench APIs.
    """
    source = OODRobustBenchSource(data_dir=data_dir)
    x, y = source.load_corruption(
        dataset=dataset,
        corruption=corruption,
        severity=severity,
        n_examples=n_examples,
        model_name=model_name,
        threat_model=threat_model,
    )
    return source.as_dataloader(x, y, batch_size=batch_size)


def get_oodrb_natural_shift_loader(
    dataset: str,
    shift: str,
    n_examples: int,
    model_name: str,
    threat_model: str = "Linf",
    batch_size: int = 64,
    data_dir: str = "./data",
):
    """
    Load OODRobustBench natural-shift data using the official OODRobustBench API.
    """
    source = OODRobustBenchSource(data_dir=data_dir)
    x, y = source.load_natural_shift(
        dataset=dataset,
        shift=shift,
        n_examples=n_examples,
        model_name=model_name,
        threat_model=threat_model,
    )
    return source.as_dataloader(x, y, batch_size=batch_size)
