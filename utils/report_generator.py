import os
import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
from fpdf import FPDF
import torch
import torchvision.transforms as transforms

class PDF(FPDF):
    def header(self):
        # Title
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Adversarial MRI Defense System Report', 0, 1, 'C')
        # Line break
        self.ln(5)
        
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        # Date
        self.cell(0, 10, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 0, 0, 'R')

def convert_image_for_report(img_source, max_size=(400, 400)):
    """
    Convert any image source to a format suitable for PDF reports.
    Handles RGBA conversion to RGB and resizing.
    Returns a BytesIO object with the image data.
    """
    try:
        # Open the image if it's a file path
        if isinstance(img_source, str) and os.path.exists(img_source):
            img = Image.open(img_source)
        elif isinstance(img_source, Image.Image):
            img = img_source
        else:
            # If it's already a BytesIO or similar object, just use it
            img = Image.open(img_source)
        
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            white_background = Image.new('RGB', img.size, (255, 255, 255))
            white_background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = white_background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize while maintaining aspect ratio
        img.thumbnail(max_size, Image.LANCZOS)
        
        # Save to BytesIO
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')  # Use PNG to maintain quality
        img_byte_arr.seek(0)  # Reset position to start of stream
        
        return img_byte_arr
    
    except Exception as e:
        print(f"Error converting image: {str(e)}")
        # Return a placeholder image in case of error
        placeholder = create_placeholder_image()
        return placeholder

def create_placeholder_image(size=(300, 300), color=(200, 200, 200)):
    """Create a simple placeholder image when actual images can't be loaded"""
    img = Image.new('RGB', size, color)
    draw = getattr(Image, 'Draw', lambda x: x)(img)
    
    # Add text if PIL.ImageDraw is available
    if hasattr(draw, 'text'):
        draw.text((size[0]//3, size[1]//2), "Image Not Available", fill=(100, 100, 100))
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr

def add_image_to_pdf(pdf, img_source, x=None, y=None, w=0, h=0, title=None):
    """Safely add an image to a PDF document with error handling"""
    try:
        # Convert the image to proper format for PDF
        img_data = convert_image_for_report(img_source)
        
        # Add title if provided
        if title:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, title, 0, 1, 'C')
        
        # Add image to PDF
        pdf.image(img_data, x=x, y=y, w=w, h=h, type='PNG')
        
        # Add some space after the image
        if y is None:  # If y is auto-positioned
            pdf.ln(h if h > 0 else 10)  # Space after image
        
        return True
    
    except Exception as e:
        # Log error and add text message instead
        print(f"Error adding image to PDF: {str(e)}")
        pdf.set_text_color(255, 0, 0)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f"Error adding image: {str(e)}", 0, 1)
        pdf.set_text_color(0, 0, 0)  # Reset text color
        pdf.ln(5)
        return False

def generate_classification_report(image_path, result_data, output_path):
    """
    Generate a PDF report for image classification results
    
    Args:
        image_path (str): Path to the original image
        result_data (dict): Classification results
        output_path (str): Path to save the PDF report
    """
    try:
        # Create PDF object
        pdf = PDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'MRI Adversarial Attack Classification Report', 0, 1, 'C')
        pdf.ln(5)
        
        # Timestamp
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
        pdf.ln(5)
        
        # Image filename
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"Image: {result_data['filename']}", 0, 1)
        pdf.ln(5)
        
        # Add original image with error handling
        add_image_to_pdf(pdf, image_path, x=55, w=100, title="Original Image")
        
        # Classification results
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Classification Results", 0, 1)
        pdf.ln(5)
        
        # Predicted class
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(60, 10, "Predicted Class:", 0, 0)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, result_data['predicted_class'], 0, 1)
        pdf.ln(5)
        
        # Confidence scores
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Confidence Scores:", 0, 1)
        
        pdf.set_font('Arial', '', 10)
        for i, (class_name, prob) in enumerate(zip(result_data['class_names'], result_data['probabilities'])):
            pdf.cell(40, 10, class_name, 1, 0, 'L')
            pdf.cell(40, 10, f"{prob*100:.2f}%", 1, 1, 'R')
        
        pdf.ln(10)
        
        # Create bar chart for probabilities
        try:
            plt.figure(figsize=(8, 4))
            plt.bar(result_data['class_names'], result_data['probabilities'], color='skyblue')
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.title('Classification Probability Distribution')
            plt.ylim(0, 1)
            
            # Save plot to a temporary BytesIO object
            plot_byte_arr = io.BytesIO()
            plt.savefig(plot_byte_arr, format='PNG')
            plt.close()
            plot_byte_arr.seek(0)  # Reset position to start of stream
            
            # Add plot to PDF
            add_image_to_pdf(pdf, plot_byte_arr, w=170)
        except Exception as e:
            pdf.cell(0, 10, f"Error generating probability chart: {str(e)}", 0, 1)
        
        # Add model contribution chart if available
        if 'attention_weights' in result_data:
            try:
                # Create bar chart for model contributions
                model_names = ['Transformer', 'CNN', 'SNN']
                contributions = result_data['attention_weights']
                
                plt.figure(figsize=(8, 4))
                bars = plt.bar(model_names, contributions, color=['#FF9999', '#66B2FF', '#99FF99'])
                
                # Add value labels above bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
                
                # Highlight the model with highest contribution
                highest_idx = np.argmax(contributions)
                bars[highest_idx].set_color('gold')
                bars[highest_idx].set_edgecolor('black')
                
                plt.xlabel('Model Component')
                plt.ylabel('Contribution Weight')
                plt.title('Contribution of Each Model Component to Classification')
                plt.ylim(0, max(contributions) + 0.2)
                
                # Add a caption for the highest contributor
                plt.figtext(0.5, 0.01, f'Highest Contribution: {model_names[highest_idx]} ({contributions[highest_idx]:.2f})',
                           ha='center', fontsize=12, fontweight='bold')
                
                # Save plot to a temporary BytesIO object
                contrib_plot = io.BytesIO()
                plt.savefig(contrib_plot, format='PNG')
                plt.close()
                contrib_plot.seek(0)  # Reset position to start of stream
                
                # Add contribution plot to PDF with explicit type
                add_image_to_pdf(pdf, contrib_plot, w=170, title="Model Contributions")
                
                # Add explanation of model contributions
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, "Model Contribution Analysis:", 0, 1)
                
                pdf.set_font('Arial', '', 10)
                contribution_explanation = (
                    f"The classification was primarily driven by the {model_names[highest_idx]} component "
                    f"with a contribution weight of {contributions[highest_idx]:.2f}. This indicates that "
                    f"the {model_names[highest_idx]} features were most informative for this particular sample."
                )
                pdf.multi_cell(0, 10, contribution_explanation)
                
            except Exception as e:
                pdf.cell(0, 10, f"Error generating model contribution chart: {str(e)}", 0, 1)
        
        # Add interpretation
        pdf.ln(10)  # Space after previous content
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Interpretation:", 0, 1)
        
        pdf.set_font('Arial', '', 10)
        class_explanations = {
            'Clean': "The image appears to be a clean MRI scan without signs of adversarial perturbation.",
            'FGSM': "The image shows signs of Fast Gradient Sign Method attack, which is a one-step attack that perturbs the image in the direction of the gradient.",
            'BIM': "The image shows signs of Basic Iterative Method attack, which is an iterative version of FGSM that applies smaller changes over multiple iterations.",
            'PGD': "The image shows signs of Projected Gradient Descent attack, which is an iterative attack with random initialization that typically creates stronger adversarial examples."
        }
        
        pdf.multi_cell(0, 10, class_explanations.get(result_data['predicted_class'], 
                                                "This class represents an unknown type of image."))
        
        # Add recommendations
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Recommendations:", 0, 1)
        
        pdf.set_font('Arial', '', 10)
        if result_data['predicted_class'] == 'Clean':
            recommendation = "No action needed. The image appears to be free of adversarial perturbations."
        else:
            recommendation = "Consider running the image through our purification system to remove adversarial perturbations."
        
        pdf.multi_cell(0, 10, recommendation)
        
        # Save PDF
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pdf.output(output_path)
            print(f"Successfully generated classification report at {output_path}")
        except Exception as e:
            print(f"Error saving PDF report: {str(e)}")
            
        return output_path
    except Exception as e:
        print(f"Error in generate_classification_report: {str(e)}")
        return None

def generate_purification_report(original_path, purified_path, comparison_path, metrics, output_path):
    """
    Generate a PDF report for image purification results
    
    Args:
        original_path (str): Path to the original image
        purified_path (str): Path to the purified image
        comparison_path (str): Path to the comparison visualization
        metrics (dict): Metrics like PSNR and SSIM
        output_path (str): Path to save the PDF report
    """
    try:
        # Create PDF object
        pdf = PDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'MRI Adversarial Purification Report', 0, 1, 'C')
        pdf.ln(5)
        
        # Timestamp
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
        pdf.ln(5)
        
        # Image filename
        pdf.set_font('Arial', 'B', 12)
        if isinstance(original_path, str) and os.path.exists(original_path):
            pdf.cell(0, 10, f"Image: {os.path.basename(original_path)}", 0, 1)
        else:
            pdf.cell(0, 10, "Original image", 0, 1)
        pdf.ln(5)
        
        # Add comparison image
        add_image_to_pdf(
            pdf, 
            comparison_path, 
            w=190, 
            title="Comparison: Original (left) vs. Purified (right)"
        )
        
        # Purification metrics
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Purification Metrics", 0, 1)
        pdf.ln(5)
        
        # PSNR
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(60, 10, "PSNR:", 0, 0)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"{metrics['psnr']:.2f} dB", 0, 1)
        
        # SSIM
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(60, 10, "SSIM:", 0, 0)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"{metrics['ssim']:.4f}", 0, 1)
        pdf.ln(5)
        
        # Interpretation of metrics
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Interpretation of Metrics:", 0, 1)
        
        pdf.set_font('Arial', '', 10)
        psnr_text = f"PSNR (Peak Signal-to-Noise Ratio): {metrics['psnr']:.2f} dB\n"
        if metrics['psnr'] > 30:
            psnr_text += "This indicates excellent image quality after purification. Higher PSNR values indicate better quality."
        elif metrics['psnr'] > 20:
            psnr_text += "This indicates good image quality after purification. PSNR values above 20 dB typically indicate acceptable quality."
        else:
            psnr_text += "This indicates moderate image quality after purification. There may be visible differences from the original."
        
        pdf.multi_cell(0, 10, psnr_text)
        pdf.ln(5)
        
        ssim_text = f"SSIM (Structural Similarity Index): {metrics['ssim']:.4f}\n"
        if metrics['ssim'] > 0.9:
            ssim_text += "This indicates excellent structural preservation. SSIM ranges from 0 to 1, with values close to 1 indicating high similarity."
        elif metrics['ssim'] > 0.7:
            ssim_text += "This indicates good structural preservation. SSIM values above 0.7 typically indicate good perceptual quality."
        else:
            ssim_text += "This indicates moderate structural preservation. There may be noticeable structural differences from the original."
        
        pdf.multi_cell(0, 10, ssim_text)
        pdf.ln(5)
        
        # Technical details about the purification model
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Technical Details", 0, 1)
        pdf.ln(3)
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Purification Model:", 0, 1)
        
        pdf.set_font('Arial', '', 10)
        model_details = (
            "This purification was performed using a U-Net based deep learning model with attention mechanisms. "
            "The model was trained in two phases:\n\n"
            "1. Phase 1: Initial training with pixel-wise and perceptual losses to learn the mapping from adversarial to clean images.\n\n"
            "2. Phase 2: GAN-based fine-tuning with a PatchGAN discriminator for improved sharpness and detail preservation.\n\n"
            "The model incorporates self-attention mechanisms, residual connections, and multi-scale feature fusion to effectively remove adversarial perturbations while preserving important diagnostic features."
        )
        
        pdf.multi_cell(0, 10, model_details)
        
        # Save PDF
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pdf.output(output_path)
            print(f"Successfully generated purification report at {output_path}")
        except Exception as e:
            print(f"Error saving PDF report: {str(e)}")
            
        return output_path
    except Exception as e:
        print(f"Error in generate_purification_report: {str(e)}")
        return None

def generate_attack_report(original_path, adversarial_path, comparison_path, 
                          attack_params, metrics, classification_results, output_path):
    """
    Generate a PDF report for adversarial attack results
    
    Args:
        original_path (str): Path to the original image
        adversarial_path (str): Path to the adversarial image
        comparison_path (str): Path to the comparison visualization
        attack_params (dict): Attack parameters (type, epsilon, etc.)
        metrics (dict): Attack metrics (L2 distance, Linf distance)
        classification_results (dict): Results of classifying the adversarial image
        output_path (str): Path to save the PDF report
    """
    try:
        # Create PDF object
        pdf = PDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'MRI Adversarial Attack Generation Report', 0, 1, 'C')
        pdf.ln(5)
        
        # Timestamp
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
        pdf.ln(5)
        
        # Image filename
        pdf.set_font('Arial', 'B', 12)
        if isinstance(original_path, str) and os.path.exists(original_path):
            pdf.cell(0, 10, f"Original Image: {os.path.basename(original_path)}", 0, 1)
        else:
            pdf.cell(0, 10, "Original image", 0, 1)
        pdf.ln(5)
        
        # Add comparison image
        add_image_to_pdf(
            pdf, 
            comparison_path, 
            w=190, 
            title=f"Comparison: Original (left) vs. {attack_params['attack_type'].upper()} (right)"
        )
        
        # Attack details
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Attack Details", 0, 1)
        pdf.ln(5)
        
        # Attack type
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(60, 10, "Attack Type:", 0, 0)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, attack_params['attack_type'].upper(), 0, 1)
        
        # Epsilon
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(60, 10, "Epsilon (ε):", 0, 0)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"{attack_params['epsilon']:.4f}", 0, 1)
        pdf.ln(5)
        
        # Attack metrics
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Attack Metrics", 0, 1)
        pdf.ln(5)
        
        # L2 distance
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(60, 10, "L2 Distance:", 0, 0)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"{metrics['l2_distance']:.4f}", 0, 1)
        
        # Linf distance
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(60, 10, "L∞ Distance:", 0, 0)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"{metrics['linf_distance']:.4f}", 0, 1)
        pdf.ln(5)
        
        # Classification results of adversarial image
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Adversarial Image Classification", 0, 1)
        pdf.ln(5)
        
        # Predicted class
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(60, 10, "Predicted Class:", 0, 0)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, classification_results['predicted_class'], 0, 1)
        pdf.ln(5)
        
        # Confidence scores
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Confidence Scores:", 0, 1)
        
        pdf.set_font('Arial', '', 10)
        for i, (class_name, prob) in enumerate(zip(classification_results['class_names'], 
                                                classification_results['probabilities'])):
            pdf.cell(40, 10, class_name, 1, 0, 'L')
            pdf.cell(40, 10, f"{prob*100:.2f}%", 1, 1, 'R')
        
        pdf.ln(10)
        
        # Create bar chart for classification probabilities
        try:
            plt.figure(figsize=(8, 4))
            plt.bar(classification_results['class_names'], classification_results['probabilities'], color='skyblue')
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.title('Adversarial Image Classification Results')
            plt.ylim(0, 1)
            
            # Save plot to a temporary BytesIO object
            plot_byte_arr = io.BytesIO()
            plt.savefig(plot_byte_arr, format='PNG')
            plt.close()
            plot_byte_arr.seek(0)  # Reset position to start of stream
            
            # Add plot to PDF
            add_image_to_pdf(pdf, plot_byte_arr, w=170)
        except Exception as e:
            pdf.cell(0, 10, f"Error generating probability chart: {str(e)}", 0, 1)
        
        # Add model contribution chart if available
        if 'attention_weights' in classification_results:
            try:
                # Create bar chart for model contributions
                model_names = ['Transformer', 'CNN', 'SNN']
                contributions = classification_results['attention_weights']
                
                plt.figure(figsize=(8, 4))
                bars = plt.bar(model_names, contributions, color=['#FF9999', '#66B2FF', '#99FF99'])
                
                # Add value labels above bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
                
                # Highlight the model with highest contribution
                highest_idx = np.argmax(contributions)
                bars[highest_idx].set_color('gold')
                bars[highest_idx].set_edgecolor('black')
                
                plt.xlabel('Model Component')
                plt.ylabel('Contribution Weight')
                plt.title('Contribution to Adversarial Classification')
                plt.ylim(0, max(contributions) + 0.2)
                
                # Add a caption for the highest contributor
                plt.figtext(0.5, 0.01, f'Highest Contribution: {model_names[highest_idx]} ({contributions[highest_idx]:.2f})',
                           ha='center', fontsize=12, fontweight='bold')
                
                # Save plot to a temporary BytesIO object
                contrib_plot = io.BytesIO()
                plt.savefig(contrib_plot, format='PNG')
                plt.close()
                contrib_plot.seek(0)  # Reset position to start of stream
                
                # Add contribution plot to PDF with explicit type
                add_image_to_pdf(pdf, contrib_plot, w=170, title="Model Contributions")
                
                # Add explanation of model contributions
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, "Model Contribution Analysis:", 0, 1)
                
                pdf.set_font('Arial', '', 10)
                contribution_explanation = (
                    f"The classification of this adversarial image was primarily driven by the {model_names[highest_idx]} component "
                    f"with a contribution weight of {contributions[highest_idx]:.2f}. This indicates that "
                    f"the {model_names[highest_idx]} features were most informative for detecting this particular attack."
                )
                pdf.multi_cell(0, 10, contribution_explanation)
                
            except Exception as e:
                pdf.cell(0, 10, f"Error generating model contribution chart: {str(e)}", 0, 1)
        
        pdf.ln(10)  # Space after chart
        
        # Description of the attack
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Attack Description", 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        attack_descriptions = {
            'fgsm': (
                "Fast Gradient Sign Method (FGSM) is a one-step attack that generates adversarial examples by perturbing the input "
                "in the direction of the gradient of the loss function with respect to the input. It's a fast but relatively "
                "simple attack that often produces noticeable perturbations."
            ),
            'bim': (
                "Basic Iterative Method (BIM) is an iterative version of FGSM that applies smaller changes over multiple iterations. "
                "This typically results in more effective adversarial examples with less visible perturbation compared to FGSM."
            ),
            'pgd': (
                "Projected Gradient Descent (PGD) is considered one of the strongest first-order adversarial attacks. "
                "It works by starting from a random point within the allowed perturbation range and then iteratively "
                "taking gradient steps, projecting back onto the allowed perturbation range after each step. "
                "This typically creates stronger adversarial examples than FGSM or BIM."
            )
        }
        
        pdf.multi_cell(0, 10, attack_descriptions.get(attack_params['attack_type'], 
                                                "No description available for this attack type."))
        
        # Save PDF
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pdf.output(output_path)
            print(f"Successfully generated attack report at {output_path}")
        except Exception as e:
            print(f"Error saving PDF report: {str(e)}")
            
        return output_path
    except Exception as e:
        print(f"Error in generate_attack_report: {str(e)}")
        return None