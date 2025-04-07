import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load surrogate model for generating adversarial examples
def load_surrogate_model():
    """Load a pre-trained model to use as surrogate for generating attacks"""
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    return model.to(device)

# Cache the surrogate model for reuse
_surrogate_model = None

def get_surrogate_model():
    """Get a cached instance of the surrogate model"""
    global _surrogate_model
    if _surrogate_model is None:
        _surrogate_model = load_surrogate_model()
    return _surrogate_model

# FGSM Attack
def fgsm_attack(image, epsilon=0.03):
    """
    Fast Gradient Sign Method attack
    
    Args:
        image (torch.Tensor): Input image tensor
        epsilon (float): Perturbation size
        
    Returns:
        torch.Tensor: Adversarial example
    """
    # Get surrogate model
    model = get_surrogate_model()
    
    # Make a copy of the input that requires gradients
    perturbed_image = image.clone().detach().requires_grad_(True)
    
    # Forward pass
    output = model(perturbed_image)
    
    # Create random target (for targeted attack) or use original prediction (for untargeted)
    target = torch.randint(0, 1000, (perturbed_image.size(0),)).to(device)
    
    # Calculate loss
    loss = F.cross_entropy(output, target)
    
    # Backward pass
    loss.backward()
    
    # Create perturbation with sign of gradient
    with torch.no_grad():
        perturbation = epsilon * perturbed_image.grad.sign()
        
        # Apply perturbation
        perturbed_image = perturbed_image + perturbation
        
        # Clamp to ensure valid pixel values
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image

# BIM Attack
def bim_attack(image, epsilon=0.03, alpha=0.01, iterations=10):
    """
    Basic Iterative Method attack
    
    Args:
        image (torch.Tensor): Input image tensor
        epsilon (float): Maximum perturbation size
        alpha (float): Step size for each iteration
        iterations (int): Number of iterations
        
    Returns:
        torch.Tensor: Adversarial example
    """
    # Get surrogate model
    model = get_surrogate_model()
    
    # Initialize perturbed image as a copy of the input
    perturbed_image = image.clone().detach()
    
    for i in range(iterations):
        # Require gradients for the perturbed image
        perturbed_image.requires_grad = True
        
        # Forward pass
        output = model(perturbed_image)
        
        # Create random target (for targeted attack)
        target = torch.randint(0, 1000, (perturbed_image.size(0),)).to(device)
        
        # Calculate loss
        loss = F.cross_entropy(output, target)
        
        # Zero gradients from previous iteration
        if perturbed_image.grad is not None:
            perturbed_image.grad.data.zero_()
            
        # Backward pass
        loss.backward()
        
        # Create perturbation with sign of gradient
        with torch.no_grad():
            # Apply small step in direction of gradient
            adv_step = alpha * perturbed_image.grad.sign()
            perturbed_image = perturbed_image + adv_step
            
            # Project back onto epsilon ball around original image
            delta = torch.clamp(perturbed_image - image, -epsilon, epsilon)
            perturbed_image = torch.clamp(image + delta, 0, 1)
        
        # Clear gradients
        perturbed_image = perturbed_image.detach()
    
    return perturbed_image

# PGD Attack
def pgd_attack(image, epsilon=0.03, alpha=0.01, iterations=20):
    """
    Projected Gradient Descent attack
    
    Args:
        image (torch.Tensor): Input image tensor
        epsilon (float): Maximum perturbation size
        alpha (float): Step size for each iteration
        iterations (int): Number of iterations
        
    Returns:
        torch.Tensor: Adversarial example
    """
    # Get surrogate model
    model = get_surrogate_model()
    
    # Initialize with random perturbation within epsilon ball
    delta = torch.rand_like(image) * 2 * epsilon - epsilon
    delta = torch.clamp(image + delta, 0, 1) - image
    
    # Start from perturbed image
    perturbed_image = (image + delta).detach()
    
    for i in range(iterations):
        # Require gradients for the perturbed image
        perturbed_image.requires_grad = True
        
        # Forward pass
        output = model(perturbed_image)
        
        # Create random target (for targeted attack) or use original prediction (for untargeted)
        target = torch.randint(0, 1000, (perturbed_image.size(0),)).to(device)
        
        # Calculate loss
        loss = F.cross_entropy(output, target)
        
        # Zero gradients from previous iteration
        if perturbed_image.grad is not None:
            perturbed_image.grad.data.zero_()
            
        # Backward pass
        loss.backward()
        
        # Create perturbation with sign of gradient
        with torch.no_grad():
            # Apply step in direction of gradient
            adv_step = alpha * perturbed_image.grad.sign()
            perturbed_image = perturbed_image + adv_step
            
            # Project back onto epsilon ball around original image
            delta = torch.clamp(perturbed_image - image, -epsilon, epsilon)
            perturbed_image = torch.clamp(image + delta, 0, 1)
        
        # Clear gradients
        perturbed_image = perturbed_image.detach()
    
    return perturbed_image

# Carlini & Wagner (C&W) Attack (simplified version)
def cw_attack(image, target=None, c=1.0, confidence=0, iterations=100, lr=0.01):
    """
    Carlini & Wagner attack (L2 norm version)
    
    Args:
        image (torch.Tensor): Input image tensor
        target (torch.Tensor, optional): Target class. If None, uses least-likely class
        c (float): Constant trading off perturbation size and loss function
        confidence (float): Confidence parameter for targeted misclassification
        iterations (int): Number of iterations
        lr (float): Learning rate for optimizer
        
    Returns:
        torch.Tensor: Adversarial example
    """
    # Get surrogate model
    model = get_surrogate_model()
    
    # If no target provided, use least-likely class
    if target is None:
        with torch.no_grad():
            output = model(image)
            _, target = torch.min(output, 1)
    
    # Convert to tanh-space for unbounded optimization
    w = torch.zeros_like(image, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=lr)
    
    # Original image in tanh-space
    image_tanh = torch.atanh(image * 2 - 1)
    
    for i in range(iterations):
        # Perturbed image in input-space
        perturbed_tanh = image_tanh + w
        perturbed_image = (torch.tanh(perturbed_tanh) + 1) / 2
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(perturbed_image)
        
        # Calculate f(x') - f(target) + confidence
        target_log_prob = torch.gather(output, 1, target.unsqueeze(1)).squeeze()
        other_log_prob = torch.max(output * (1 - F.one_hot(target, 1000)), dim=1)[0]
        f_loss = torch.clamp(other_log_prob - target_log_prob + confidence, min=0)
        
        # Calculate L2 distance
        l2_dist = torch.sum((perturbed_image - image) ** 2, dim=[1, 2, 3])
        
        # Total loss
        loss = torch.mean(l2_dist + c * f_loss)
        
        # Backward pass
        loss.backward()
        
        # Update perturbation
        optimizer.step()
        
        # Clip perturbation if needed (for attack budget)
        with torch.no_grad():
            delta = perturbed_image - image
            norm = torch.norm(delta.view(delta.shape[0], -1), dim=1)
            factor = torch.min(torch.ones_like(norm), 1.0 / norm)
            perturbed_image = torch.clamp(image + delta * factor.view(-1, 1, 1, 1), 0, 1)
    
    return perturbed_image.detach()

# DeepFool Attack (simplified version)
def deepfool_attack(image, epsilon=0.1, max_iter=50):
    """
    DeepFool attack
    
    Args:
        image (torch.Tensor): Input image tensor
        epsilon (float): Maximum perturbation size
        max_iter (int): Maximum number of iterations
        
    Returns:
        torch.Tensor: Adversarial example
    """
    # Get surrogate model
    model = get_surrogate_model()
    
    perturbed_image = image.clone().detach()
    perturbed_image.requires_grad = True
    
    with torch.no_grad():
        output = model(image)
        original_pred = torch.argmax(output, dim=1)
    
    for i in range(max_iter):
        if perturbed_image.grad is not None:
            perturbed_image.grad.data.zero_()
            
        # Forward pass
        output = model(perturbed_image)
        
        # Get current prediction
        pred = torch.argmax(output, dim=1)
        
        # If already adversarial, stop
        if pred != original_pred:
            break
            
        # Calculate loss and gradients
        loss = F.cross_entropy(output, original_pred)
        loss.backward()
        
        # Get gradient
        grad = perturbed_image.grad.data
        
        # Compute perturbation
        with torch.no_grad():
            # Simple perturbation in gradient direction
            perturb = epsilon * grad / (torch.norm(grad) + 1e-7)
            perturbed_image = torch.clamp(perturbed_image + perturb, 0, 1)
            
        # Detach for next iteration
        perturbed_image = perturbed_image.detach().requires_grad_(True)
    
    return perturbed_image.detach()