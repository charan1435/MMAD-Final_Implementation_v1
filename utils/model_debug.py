import logging
import os
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_debug.log')
    ]
)
logger = logging.getLogger('model_debug')

def save_debug_image(img_tensor, filename="debug_image.png"):
    """Save tensor as image for debugging"""
    if not os.path.exists('debug'):
        os.makedirs('debug')
    
    if img_tensor.ndim == 4:
        img_tensor = img_tensor.squeeze(0)
    
    # Convert to CPU, detach from computation graph, and ensure it's in [0,1] range
    img_tensor = img_tensor.cpu().detach().clamp(0, 1)
    
    # Convert to PIL and save
    img = transforms.ToPILImage()(img_tensor)
    save_path = os.path.join('debug', filename)
    img.save(save_path)
    logger.info(f"Saved debug image to {save_path}")
    return save_path

def analyze_model_predictions(model, image_tensor, class_names=['Clean', 'FGSM', 'BIM', 'PGD']):
    """Analyze model predictions for debugging"""
    model.eval()  # Ensure model is in eval mode
    
    # Create a timestamp for this analysis
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Get model output
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Process outputs
    if isinstance(outputs, dict):
        logits = outputs['logits']
        attention_weights = outputs.get('attention_weights', None)
        aux_logits = outputs.get('aux_logits', None)
    else:
        logits = outputs
        attention_weights = None
        aux_logits = None
    
    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
    
    # Log results
    logger.info(f"Model prediction analysis ({timestamp}):")
    logger.info(f"Raw logits: {logits.cpu().numpy()}")
    
    # Log probabilities for each class
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        logger.info(f"  {class_name}: {prob.item()*100:.2f}%")
    
    # Log prediction and confidence
    max_prob, prediction = torch.max(probabilities, dim=0)
    logger.info(f"Main prediction: {class_names[prediction.item()]} with confidence: {max_prob.item()*100:.2f}%")
    
    # Log individual component predictions if available
    if aux_logits is not None:
        component_names = ['transformer', 'cnn', 'snn']
        logger.info("Individual component predictions:")
        
        for comp_name in component_names:
            if comp_name in aux_logits:
                comp_logits = aux_logits[comp_name]
                comp_probs = torch.nn.functional.softmax(comp_logits, dim=1)[0]
                comp_pred = torch.argmax(comp_probs, dim=0).item()
                comp_conf = comp_probs[comp_pred].item()
                logger.info(f"  {comp_name}: {class_names[comp_pred]} ({comp_conf*100:.2f}%)")
                
                # Save component predictions for return
                if 'component_predictions' not in locals():
                    component_predictions = {}
                component_predictions[comp_name] = {
                    'class': class_names[comp_pred],
                    'confidence': comp_conf,
                    'probabilities': comp_probs.cpu().numpy().tolist()
                }
    
    # Log attention weights if available
    if attention_weights is not None:
        logger.info(f"Attention weights: {attention_weights.cpu().numpy()}")
        
        # Create bar chart of component contributions
        model_components = ['Vision Transformer', 'CNN', 'SNN']
        chart_path = create_contribution_chart(
            attention_weights.cpu().numpy()[0], 
            model_components,
            f'debug/contributions_{timestamp}.png'
        )
        logger.info(f"Saved contribution chart to {chart_path}")
    
    # Also save input image for reference
    if image_tensor is not None:
        # For normalized images, we need to denormalize
        # Assuming ImageNet normalization (adjust if different)
        denorm = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
        ])
        
        try:
            # Try to denormalize (will fail if not normalized)
            denormalized = denorm(image_tensor[0])
            input_path = save_debug_image(denormalized.unsqueeze(0), f"input_{timestamp}.png")
        except:
            # If denormalization fails, save as is
            input_path = save_debug_image(image_tensor, f"input_{timestamp}.png")
    
    result = {
        'timestamp': timestamp,
        'logits': logits.cpu().numpy().tolist(),
        'probabilities': probabilities.cpu().numpy().tolist(),
        'prediction': prediction.item(),
        'predicted_class': class_names[prediction.item()],
        'confidence': max_prob.item(),
    }
    
    # Add optional data if available
    if attention_weights is not None:
        result['attention_weights'] = attention_weights.cpu().numpy()[0].tolist()
    
    if 'component_predictions' in locals():
        result['component_predictions'] = component_predictions
    
    return result

def create_contribution_chart(attention_weights, component_names, save_path='debug/contributions.png'):
    """Create and save a chart showing model component contributions"""
    if not os.path.exists('debug'):
        os.makedirs('debug')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert weights to percentages
    weights_pct = attention_weights * 100
    
    # Create bar chart
    bars = ax.bar(component_names, weights_pct, color=['#8ecae6', '#219ebc', '#023047'])
    
    # Highlight the highest contribution
    highest_idx = np.argmax(attention_weights)
    bars[highest_idx].set_color('#ffb703')
    bars[highest_idx].set_edgecolor('black')
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1, 
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add title and labels
    ax.set_title('Model Component Contributions', fontsize=14, fontweight='bold')
    ax.set_ylabel('Contribution Percentage (%)')
    ax.set_ylim(0, max(weights_pct) * 1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add note
    fig.text(0.5, 0.01, 'Higher percentage indicates greater contribution to the classification decision', 
             ha='center', fontsize=10, fontstyle='italic')
    
    # Save the chart
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path)
    plt.close(fig)
    
    logger.info(f"Saved contribution chart to {save_path}")
    return save_path

def analyze_multiple_images(model, image_dir, transform=None, device='cpu'):
    """Analyze multiple images to understand patterns in classification"""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    results = []
    if not os.path.exists(image_dir):
        logger.error(f"Directory not found: {image_dir}")
        return results
        
    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            try:
                img_path = os.path.join(image_dir, img_file)
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Analyze this image
                result = analyze_model_predictions(model, image_tensor)
                result['filename'] = img_file
                results.append(result)
                
                # Save analysis to log
                logger.info(f"Analyzed {img_file}: predicted as {result['predicted_class']} with confidence {result['confidence']*100:.2f}%")
                
            except Exception as e:
                logger.error(f"Error analyzing {img_file}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
    
    # Generate summary statistics
    if results:
        logger.info("\nSummary Statistics:")
        class_counts = {i: 0 for i in range(4)}
        total_confidence = {i: 0.0 for i in range(4)}
        
        for result in results:
            pred_idx = result['prediction']
            class_counts[pred_idx] += 1
            total_confidence[pred_idx] += result['confidence']
        
        total = len(results)
        logger.info(f"Analyzed {total} images:")
        
        for class_idx, count in class_counts.items():
            if count > 0:
                class_name = ['Clean', 'FGSM', 'BIM', 'PGD'][class_idx]
                percentage = (count / total) * 100 if total > 0 else 0
                avg_confidence = total_confidence[class_idx] / count if count > 0 else 0
                logger.info(f"  {class_name}: {count}/{total} ({percentage:.1f}%) - Avg confidence: {avg_confidence*100:.1f}%")
    
        # Create summary graph
        create_classification_summary(class_counts, total)
    
    return results

def create_classification_summary(class_counts, total, save_path='debug/classification_summary.png'):
    """Create a summary chart of classification results"""
    if not os.path.exists('debug'):
        os.makedirs('debug')
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    class_names = ['Clean', 'FGSM', 'BIM', 'PGD']
    counts = [class_counts[i] for i in range(4)]
    percentages = [(count / total) * 100 if total > 0 else 0 for count in counts]
    
    # Colors for each class
    colors = ['#2a9d8f', '#e63946', '#f4a261', '#457b9d']
    
    # Create bar chart
    bars = ax.bar(class_names, percentages, color=colors)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 1, 
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add title and labels
    ax.set_title('Classification Distribution Summary', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage of Images (%)')
    ax.set_ylim(0, max(percentages) * 1.2 if max(percentages) > 0 else 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add total count
    fig.text(0.5, 0.01, f'Total images analyzed: {total}', 
             ha='center', fontsize=10, fontstyle='italic')
    
    # Save the chart
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path)
    plt.close(fig)
    
    logger.info(f"Saved classification summary to {save_path}")
    return save_path

def inspect_model(model):
    """Inspect model architecture and weights for debugging"""
    logger.info("Model inspection:")
    
    # Log model architecture
    logger.info(f"Model structure:\n{model}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Check for NaNs in model weights
    has_nan = False
    has_large_weights = False
    has_tiny_weights = False
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            has_nan = True
            logger.error(f"NaN detected in {name}")
        
        max_val = param.abs().max().item()
        
        if max_val > 100:
            has_large_weights = True
            logger.warning(f"Large weight detected in {name}: max abs value = {max_val}")
        
        if param.numel() > 0 and max_val < 1e-6:
            has_tiny_weights = True
            logger.warning(f"Very small weight detected in {name}: max abs value = {max_val}")
    
    if not has_nan:
        logger.info("No NaNs detected in model weights")
    
    # Log mean and std of weights for key layers
    logger.info("Weight statistics for key layers:")
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            logger.info(f"  {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}, "
                       f"min={param.min().item():.6f}, max={param.max().item():.6f}")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'has_nan': has_nan,
        'has_large_weights': has_large_weights,
        'has_tiny_weights': has_tiny_weights
    }

def check_normalization(image_tensor):
    """Check if an image tensor appears to be normalized"""
    # Unnormalized images typically have values between 0 and 1
    # Normalized images (with ImageNet norms) will have negative values and values > 1
    
    if image_tensor.min() < 0 or image_tensor.max() > 1.1:
        logger.info("Image appears to be normalized (values outside [0,1] range)")
        return True
    else:
        logger.info("Image appears to be unnormalized (values within [0,1] range)")
        return False

def compare_model_versions(model1, model2, device='cpu'):
    """Compare two versions of the same model architecture"""
    logger.info("Comparing model versions:")
    
    # Put both models in eval mode
    model1.eval()
    model2.eval()
    
    # Check parameter counts
    m1_params = sum(p.numel() for p in model1.parameters())
    m2_params = sum(p.numel() for p in model2.parameters())
    
    logger.info(f"Parameter counts - Model 1: {m1_params:,}, Model 2: {m2_params:,}")
    
    # Check if parameters match
    if m1_params != m2_params:
        logger.warning("Models have different parameter counts - architectures likely differ")
        return {
            'architectures_match': False,
            'param_count_model1': m1_params,
            'param_count_model2': m2_params
        }
    
    # Compare weights of matching parameters
    max_diff = 0
    diff_layers = []
    matching_layers = 0
    total_layers = 0
    
    for (name1, p1), (name2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            logger.warning(f"Layer name mismatch: {name1} vs {name2}")
            continue
        
        total_layers += 1
        
        if p1.shape != p2.shape:
            logger.warning(f"Shape mismatch in layer {name1}: {p1.shape} vs {p2.shape}")
            continue
        
        # Compute differences
        p1_data = p1.data.to(device)
        p2_data = p2.data.to(device)
        
        # Calculate difference
        diff = torch.abs(p1_data - p2_data).max().item()
        
        if diff < 1e-6:
            matching_layers += 1
        else:
            max_diff = max(max_diff, diff)
            diff_layers.append((name1, diff))
    
    # Sort and log the layers with the most difference
    diff_layers.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"Models have {matching_layers}/{total_layers} matching layers")
    
    if diff_layers:
        logger.info("Top 5 layers with most difference:")
        for name, diff in diff_layers[:5]:
            logger.info(f"  {name}: max difference = {diff}")
    
    return {
        'architectures_match': m1_params == m2_params,
        'matching_layers': matching_layers,
        'total_layers': total_layers,
        'max_difference': max_diff,
        'different_layers': diff_layers
    }

def test_model_with_sample_input(model, device='cpu'):
    """Test model with a simple gradient-filled sample input"""
    logger.info("Testing model with sample input...")
    
    # Create a sample input (gradient pattern)
    x = torch.linspace(0, 1, 224).view(1, 1, 224, 1).expand(1, 3, 224, 224).to(device)
    y = torch.linspace(0, 1, 224).view(1, 1, 1, 224).expand(1, 3, 224, 224).to(device)
    
    sample_input = (x + y) / 2.0
    
    # Save sample image
    save_debug_image(sample_input, "sample_input.png")
    
    # Test both normalized and unnormalized
    results = {}
    
    # Unnormalized
    try:
        model.eval()
        with torch.no_grad():
            output_unnorm = model(sample_input)
        
        if isinstance(output_unnorm, dict):
            logits_unnorm = output_unnorm['logits']
        else:
            logits_unnorm = output_unnorm
            
        probs_unnorm = torch.nn.functional.softmax(logits_unnorm, dim=1)[0]
        pred_unnorm = torch.argmax(probs_unnorm, dim=0).item()
        conf_unnorm = probs_unnorm[pred_unnorm].item()
        
        results['unnormalized'] = {
            'prediction': pred_unnorm,
            'confidence': conf_unnorm,
            'probabilities': probs_unnorm.cpu().numpy().tolist()
        }
        
        logger.info(f"Unnormalized input - Prediction: {pred_unnorm}, Confidence: {conf_unnorm*100:.2f}%")
    except Exception as e:
        logger.error(f"Error testing unnormalized input: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Normalized (ImageNet norms)
    try:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        normalized_input = normalize(sample_input.squeeze(0)).unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            output_norm = model(normalized_input)
        
        if isinstance(output_norm, dict):
            logits_norm = output_norm['logits']
        else:
            logits_norm = output_norm
            
        probs_norm = torch.nn.functional.softmax(logits_norm, dim=1)[0]
        pred_norm = torch.argmax(probs_norm, dim=0).item()
        conf_norm = probs_norm[pred_norm].item()
        
        results['normalized'] = {
            'prediction': pred_norm,
            'confidence': conf_norm,
            'probabilities': probs_norm.cpu().numpy().tolist()
        }
        
        logger.info(f"Normalized input - Prediction: {pred_norm}, Confidence: {conf_norm*100:.2f}%")
    except Exception as e:
        logger.error(f"Error testing normalized input: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    if 'normalized' in results and 'unnormalized' in results:
        if results['normalized']['prediction'] != results['unnormalized']['prediction']:
            logger.warning("Model gives different predictions for normalized vs unnormalized inputs!")
    
    return results