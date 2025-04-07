import os
import io
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for model input
    """
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def tensor_to_image(tensor):
    """
    Convert a tensor to a PIL Image
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    
    tensor = tensor.cpu().detach()
    tensor = tensor.clamp(0, 1)
    
    return transforms.ToPILImage()(tensor)

def save_image(tensor, output_path):
    """
    Save a tensor as an image
    """
    image = tensor_to_image(tensor)
    image.save(output_path)
    return output_path

def generate_grid_visualization(images, titles, output_path, figsize=(10, 5)):
    """
    Generate a grid visualization of images with titles
    """
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    
    if len(images) == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        if isinstance(img, torch.Tensor):
            img = tensor_to_image(img)
        
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two image tensors
    """
    # Ensure inputs are in [0, 1]
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    Calculate SSIM between two image tensors
    """
    # Ensure inputs are in [0, 1]
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean([1, 2, 3])

def create_heatmap(tensor, output_path):
    """
    Create a heatmap visualization of a tensor
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    
    if tensor.size(0) == 3:
        # Convert RGB to grayscale for heatmap
        tensor = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
    else:
        tensor = tensor.squeeze(0)
    
    tensor = tensor.cpu().detach().numpy()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(tensor, cmap='hot')
    plt.colorbar()
    plt.title('Perturbation Heatmap')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def create_difference_map(original_tensor, modified_tensor, output_path, amplification=5):
    """
    Create a visualization of the difference between two tensors
    """
    if original_tensor.ndim == 4:
        original_tensor = original_tensor.squeeze(0)
    if modified_tensor.ndim == 4:
        modified_tensor = modified_tensor.squeeze(0)
    
    # Calculate absolute difference
    diff = torch.abs(modified_tensor - original_tensor) * amplification
    diff = torch.clamp(diff, 0, 1)
    
    # Convert to numpy for visualization
    diff_np = diff.cpu().detach().permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(diff_np)
    plt.title('Difference Map (amplified)')
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def plot_image_histogram(tensor, output_path, bins=256):
    """
    Plot histogram of image pixel values
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    
    # Convert to numpy for histogram
    img_np = tensor.cpu().detach().permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(8, 4))
    
    # Plot histogram for each channel
    colors = ['r', 'g', 'b']
    for i, color in enumerate(colors):
        plt.hist(img_np[:,:,i].flatten(), bins=bins, color=color, alpha=0.5, 
                 label=f'Channel {i}', range=(0, 1))
    
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Image Histogram')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path