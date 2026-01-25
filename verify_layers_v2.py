
import torch
from robustbench import load_model

def print_some_layers(model_name):
    print(f"\nModules for {model_name}:")
    model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
    count = 0
    for name, _ in model.named_modules():
        if count < 50:
            if 'layer' in name:
                print(name)
                count += 1
        else:
            break

print_some_layers('Standard')
print_some_layers('Gowal2021Improving_28_10_ddpm_100m')
