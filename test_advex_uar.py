
import torch
from src.perturber import Perturber
from src.loader import get_resnet18, get_cifar10_loader
import matplotlib.pyplot as plt

def test_advex_uar():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_resnet18().to(device).eval()
    loader = get_cifar10_loader(batch_size=4)
    images, labels = next(iter(loader))
    images = images.to(device)
    
    perturber = Perturber(model=model, device=device)
    
    # Test fog attack
    print("Testing Fog Attack...")
    fog_images = perturber.add('fog', eps=16.0, n_iters=5).apply(images)
    print(f"Fog images range: {fog_images.min().item():.2f} to {fog_images.max().item():.2f}")
    
    # Test snow attack
    perturber.clear().add('snow', eps=0.1, n_iters=5)
    snow_images = perturber.apply(images)
    print(f"Snow images range: {snow_images.min().item():.2f} to {snow_images.max().item():.2f}")
    
    # Plot results
    # Save a comparison
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Clean")
    plt.imshow(images[0].cpu().permute(1, 2, 0))
    
    plt.subplot(1, 3, 2)
    plt.title("Fog")
    plt.imshow(fog_images[0].detach().cpu().permute(1, 2, 0))
    
    plt.subplot(1, 3, 3)
    plt.title("Snow")
    plt.imshow(snow_images[0].detach().cpu().permute(1, 2, 0))
    
    plt.savefig('test_advex_uar.png')
    print("Saved test_advex_uar.png")

if __name__ == "__main__":
    test_advex_uar()
