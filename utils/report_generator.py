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
        
        # Add original image
        try:
            # Make sure we're dealing with a file path, not a BytesIO object
            if isinstance(image_path, str) and os.path.exists(image_path):
                img = Image.open(image_path)
                img = img.resize((300, 300), Image.LANCZOS)
                
                # Save to a temporary BytesIO object
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Add image to PDF
                pdf.image(io.BytesIO(img_byte_arr), x=55, y=None, w=100)
                pdf.ln(110)  # Space after image
            else:
                pdf.cell(0, 10, "Unable to include image in report", 0, 1)
                pdf.ln(5)
                
        except Exception as e:
            pdf.cell(0, 10, f"Error adding image to report: {str(e)}", 0, 1)
            pdf.ln(5)
        
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
            plot_byte_arr = plot_byte_arr.getvalue()
            
            # Add plot to PDF
            pdf.image(io.BytesIO(plot_byte_arr), x=20, y=None, w=170)
        except Exception as e:
            pdf.cell(0, 10, f"Error generating probability chart: {str(e)}", 0, 1)
        
        # Add interpretation
        pdf.ln(100)  # Space after chart
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
        try:
            if isinstance(comparison_path, str) and os.path.exists(comparison_path):
                img = Image.open(comparison_path)
                img = img.resize((400, 200), Image.LANCZOS)
                # Save to a temporary BytesIO object
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Add image to PDF
                pdf.image(io.BytesIO(img_byte_arr), x=10, y=None, w=190)
                pdf.ln(100)  # Space after image
            else:
                pdf.cell(0, 10, "Comparison image not available", 0, 1)
                pdf.ln(5)
                
        except Exception as e:
            pdf.cell(0, 10, f"Error adding comparison image: {str(e)}", 0, 1)
            pdf.ln(5)
        
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