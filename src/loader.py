import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from robustbench.utils import load_model

def get_resnet18(model_name='Standard'):
    """
    完全调用 RobustBench 官方 API 加载模型
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
    RobustBench 的 load_cifar10 只提供测试集。
    这里使用 torchvision.datasets.CIFAR10 以严格区分 train / test。
    RB 模型通常自带归一化层，因此保持 [0, 1] 的 ToTensor 即可。
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
