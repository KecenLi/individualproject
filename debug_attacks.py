
import torch
import torchvision
from src.loader import get_resnet18, get_cifar10_loader
from src.perturber import Perturber
import matplotlib.pyplot as plt
import os

def debug_attacks():
    print("Initializing Model...")
    model = get_resnet18('Standard').cuda()
    model.eval()
    
    print("Loading Data...")
    loader = get_cifar10_loader(batch_size=8, n_examples=16)
    images, labels = next(iter(loader))
    images = images.cuda()
    labels = labels.cuda()
    
    perturber = Perturber(model, 'cuda')
    
    # List of attacks to debug
    attacks = {
        'Clean': [],
        'AutoAttack_L2': [('autoattack', {'norm': 'L2', 'eps': 0.5, 'version': 'fast'})],
        'Fog': [('fog', {'eps': 255.0, 'n_iters': 10})],    # Current Suspect
        'Snow': [('snow', {'eps': 0.1, 'n_iters': 10})],   # Current Suspect
        'Gabor': [('gabor', {'eps': 40.0, 'n_iters': 10})], # Current Suspect
        'JPEG_Linf': [('jpeg', {'norm': 'linf', 'eps': 0.125, 'n_iters': 10})],
    }
    
    os.makedirs("debug_vis", exist_ok=True)
    
    for name, config in attacks.items():
        print(f"Running {name}...")
        perturber.clear()
        for c_name, c_params in config:
            perturber.add(c_name, **c_params)
            
        try:
            adv_images = perturber.apply(images, labels)
            
            # Check statistics
            min_val = adv_images.min().item()
            max_val = adv_images.max().item()
            mean_val = adv_images.mean().item()
            
            # Predict
            with torch.no_grad():
                logits = model(adv_images)
                preds = logits.argmax(1)
                acc = (preds == labels).float().mean().item()
            
            print(f"  [{name}] Range: [{min_val:.4f}, {max_val:.4f}] | Mean: {mean_val:.4f} | Batch Acc: {acc:.2%}")
            
            # Save visual
            grid = torchvision.utils.make_grid(adv_images, nrow=4, normalize=True)
            torchvision.utils.save_image(grid, f"debug_vis/{name}.png")
            
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_attacks()
