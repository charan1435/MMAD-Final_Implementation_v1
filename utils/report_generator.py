import os
import datetime
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
from fpdf import FPDF
import torch
import torchvision.transforms as transforms
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('report_generator')

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

def create_placeholder_image(size=(300, 300), color=(200, 200, 200)):
    """Create a simple placeholder image when actual images can't be loaded"""
    try:
        img = Image.new('RGB', size, color)
        
        # Add text if possible
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            text = "Image Not Available"
            text_position = (size[0]//3, size[1]//2)
            draw.text(text_position, text, fill=(100, 100, 100))
        except Exception as e:
            logger.warning(f"Could not add text to placeholder image: {str(e)}")
        
        # Save to temporary file instead of BytesIO
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_file.close()
        img.save(temp_file.name, format='PNG')
        
        return temp_file.name
    except Exception as e:
        logger.error(f"Error creating placeholder image: {str(e)}")
        # Create an absolutely minimal valid PNG file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_file.close()
        with open(temp_file.name, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82')
        return temp_file.name

def convert_image_for_report(img_source, max_size=(400, 400)):
    """
    Convert any image source to a temporary file suitable for PDF reports.
    Returns a path to a temporary file.
    """
    temp_file = None
    try:
        # Handle different types of image sources
        if isinstance(img_source, str):
            # It's a file path
            if not os.path.exists(img_source):
                logger.error(f"Image file not found: {img_source}")
                return create_placeholder_image()
            try:
                img = Image.open(img_source)
            except (UnidentifiedImageError, IOError) as e:
                logger.error(f"Error opening image file: {str(e)}")
                return create_placeholder_image()
                
        elif isinstance(img_source, Image.Image):
            # It's already a PIL Image
            img = img_source
            
        elif isinstance(img_source, io.BytesIO):
            # It's a BytesIO object
            try:
                # Save position
                pos = img_source.tell()
                # Rewind to beginning
                img_source.seek(0)
                img = Image.open(img_source)
                # Return to original position after reading
                img_source.seek(pos)
            except Exception as e:
                logger.error(f"Error reading BytesIO image: {str(e)}")
                return create_placeholder_image()
                
        elif hasattr(img_source, 'read'):
            # It's a file-like object
            try:
                img = Image.open(img_source)
            except Exception as e:
                logger.error(f"Error opening file-like object as image: {str(e)}")
                return create_placeholder_image()
                
        else:
            # Unsupported type
            logger.error(f"Unsupported image source type: {type(img_source)}")
            return create_placeholder_image()
        
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            white_background = Image.new('RGB', img.size, (255, 255, 255))
            white_background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = white_background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize while maintaining aspect ratio
        img.thumbnail(max_size, Image.LANCZOS)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_file.close()
        img.save(temp_file.name, format='PNG')
        
        return temp_file.name
    
    except Exception as e:
        logger.error(f"Error converting image: {str(e)}")
        # If we've already created a temp file but encountered an error, clean it up
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass
        # Return a placeholder image in case of error
        return create_placeholder_image()

def add_image_to_pdf(pdf, img_source, x=None, y=None, w=0, h=0, title=None):
    """Safely add an image to a PDF document with error handling"""
    temp_file = None
    try:
        # Convert the image to a temporary file for PDF
        img_file = convert_image_for_report(img_source)
        temp_file = img_file  # Keep track for cleanup
        
        # Add title if provided
        if title:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, title, 0, 1, 'C')
        
        # Add image to PDF using the temporary file
        pdf.image(img_file, x=x, y=y, w=w, h=h)
        
        # Add some space after the image
        if y is None:  # If y is auto-positioned
            pdf.ln(h if h > 0 else 10)  # Space after image
        
        return True
    
    except Exception as e:
        # Log error and add text message instead
        logger.error(f"Error adding image to PDF: {str(e)}")
        pdf.set_text_color(255, 0, 0)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f"Error adding image: {str(e)}", 0, 1)
        pdf.set_text_color(0, 0, 0)  # Reset text color
        pdf.ln(5)
        return False
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file}: {e}")

def save_plot_to_temp_file(fig, close_figure=True):
    """Save a matplotlib figure to a temporary file and return the path"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    temp_file.close()
    
    try:
        fig.savefig(temp_file.name, format='PNG', bbox_inches='tight')
        if close_figure:
            plt.close(fig)
    except Exception as e:
        logger.error(f"Error saving plot to file: {str(e)}")
        try:
            os.unlink(temp_file.name)
        except:
            pass
        return None
        
    return temp_file.name

def add_model_contribution_chart(pdf, attention_weights, temp_files, report_type="classification"):
    """
    Create and add a chart showing model component contributions to the PDF
    
    Args:
        pdf: The PDF object to add the chart to
        attention_weights: List of contribution weights from each model
        temp_files: List to track temporary files for cleanup
        report_type: Type of report ("classification" or "attack")
    
    Returns:
        bool: True if chart was added successfully, False otherwise
    """
    fig = None
    try:
        # Create bar chart for model contributions
        # Use more descriptive model names
        model_names = ['Vision Transformer', 'CNN', 'SNN']
        contributions = attention_weights
        
        # Convert to percentages for clearer visualization
        contribution_percentages = [weight * 100 for weight in contributions]
        
        # Create the figure with a slightly larger size for better readability
        fig = plt.figure(figsize=(9, 5))
        
        # Create gradient colors for bars (from light to dark)
        colors = ['#8ecae6', '#219ebc', '#023047']  # Blue color palette
        
        # Plot the bars
        bars = plt.bar(model_names, contribution_percentages, color=colors, 
                       edgecolor='black', linewidth=1, alpha=0.8)
        
        # Add percentage labels above bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                     f'{height:.1f}%', ha='center', va='bottom', 
                     fontweight='bold', fontsize=10)
        
        # Highlight the model with highest contribution
        highest_idx = np.argmax(contributions)
        bars[highest_idx].set_color('#ffb703')  # Highlight color (golden)
        bars[highest_idx].set_edgecolor('black')
        bars[highest_idx].set_linewidth(2)
        
        # Set chart title based on report type
        if report_type == "attack":
            title = 'Model Component Contributions to Adversarial Detection'
        else:
            title = 'Model Component Contributions to Classification'
            
        plt.xlabel('Model Architecture Component', fontsize=11)
        plt.ylabel('Contribution Percentage (%)', fontsize=11)
        plt.title(title, fontsize=13, fontweight='bold')
        plt.ylim(0, max(contribution_percentages) + 10)  # Add some space at top
        
        # Add grid lines for easier reading of percentages
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add a box around the plot
        plt.box(True)
        
        # Add a note about how to interpret the chart
        plt.figtext(0.5, 0.01, 
                   f'Higher percentage indicates greater contribution to the final decision',
                   ha='center', fontsize=10, fontstyle='italic')
                   
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for the note
        
        # Save plot to a temporary file
        contrib_plot_file = save_plot_to_temp_file(fig)
        if contrib_plot_file:
            temp_files.append(contrib_plot_file)
            # Add contribution plot to PDF
            add_image_to_pdf(pdf, contrib_plot_file, w=180, title="Model Architecture Contribution Analysis")
        plt.close(fig)  # Ensure the figure is closed
        fig = None
        
        # Add explanation of model contributions
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Model Contribution Breakdown:", 0, 1)
        
        # Get the dominant model for explanation
        highest_contrib = contributions[highest_idx]
        dominant_model = model_names[highest_idx]
        
        # Calculate relative strength of the dominant model
        second_highest = sorted(contributions, reverse=True)[1]
        relative_strength = highest_contrib / second_highest
        
        # Descriptive text about how confident the model is
        confidence_desc = ""
        if relative_strength > 2:
            confidence_desc = "strongly dominated"
        elif relative_strength > 1.5:
            confidence_desc = "significantly influenced"
        else:
            confidence_desc = "primarily guided"
        
        pdf.set_font('Arial', '', 10)
        
        # Create a more detailed explanation
        if report_type == "attack":
            contribution_explanation = (
                f"The adversarial detection was {confidence_desc} by the {dominant_model} component "
                f"with a contribution of {highest_contrib*100:.1f}%. This indicates that this component was "
                f"most effective at identifying the characteristics of this particular attack type.\n\n"
                
                f"Contribution percentages reflect how much each architectural component influenced "
                f"the final classification decision through the model's attention-based fusion mechanism."
            )
        else:
            contribution_explanation = (
                f"The classification decision was {confidence_desc} by the {dominant_model} component "
                f"with a contribution of {highest_contrib*100:.1f}%. This indicates that the features extracted "
                f"by this component were most informative for this particular image.\n\n"
                
                f"Contribution percentages reflect how much each architectural component influenced "
                f"the final classification decision through the model's attention-based fusion mechanism."
            )
            
        pdf.multi_cell(0, 10, contribution_explanation)
        
        return True
    except Exception as e:
        logger.error(f"Error generating model contribution chart: {str(e)}")
        pdf.cell(0, 10, f"Error generating model contribution chart: {str(e)}", 0, 1)
        return False
    finally:
        # Ensure figure is closed
        if fig is not None:
            plt.close(fig)

def generate_classification_report(image_path, result_data, output_path):
    """
    Generate a PDF report for image classification results
    
    Args:
        image_path (str): Path to the original image
        result_data (dict): Classification results
        output_path (str): Path to save the PDF report
    """
    temp_files = []  # Keep track of all temporary files
    
    try:
        logger.info(f"Generating classification report for {image_path} to {output_path}")
        
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
        fig = None
        try:
            fig = plt.figure(figsize=(8, 4))
            
            # Use color coding for different classes
            colors = ['#2a9d8f', '#e63946', '#f4a261', '#457b9d']
            
            # Create the bar chart
            bars = plt.bar(result_data['class_names'], 
                   [prob * 100 for prob in result_data['probabilities']], 
                   color=colors)
                   
            # Add percentage labels above bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                         f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.xlabel('Attack Type')
            plt.ylabel('Confidence Percentage (%)')
            plt.title('Classification Confidence Distribution')
            plt.ylim(0, max([prob * 100 for prob in result_data['probabilities']]) + 10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save plot to a temporary file
            plot_file = save_plot_to_temp_file(fig)
            if plot_file:
                temp_files.append(plot_file)
                add_image_to_pdf(pdf, plot_file, w=170)
            plt.close(fig)  # Ensure the figure is closed
            fig = None
        except Exception as e:
            logger.error(f"Error generating probability chart: {str(e)}")
            pdf.cell(0, 10, f"Error generating probability chart: {str(e)}", 0, 1)
        finally:
            # Ensure figure is closed
            if fig is not None:
                plt.close(fig)
        
        # Add model contribution chart if available
        if 'attention_weights' in result_data:
            add_model_contribution_chart(pdf, result_data['attention_weights'], temp_files, "classification")
        
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
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create if there's a directory part
                os.makedirs(output_dir, exist_ok=True)
            pdf.output(output_path)
            logger.info(f"Successfully generated classification report at {output_path}")
        except Exception as e:
            logger.error(f"Error saving PDF report: {str(e)}")
            
        return output_path
    except Exception as e:
        logger.error(f"Error in generate_classification_report: {str(e)}")
        return None
    finally:
        # Clean up all temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file}: {e}")

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
    temp_files = []  # Keep track of all temporary files
    
    try:
        logger.info(f"Generating purification report for {original_path} to {output_path}")
        
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
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create if there's a directory part
                os.makedirs(output_dir, exist_ok=True)
            pdf.output(output_path)
            logger.info(f"Successfully generated purification report at {output_path}")
        except Exception as e:
            logger.error(f"Error saving PDF report: {str(e)}")
            
        return output_path
    except Exception as e:
        logger.error(f"Error in generate_purification_report: {str(e)}")
        return None
    finally:
        # Clean up all temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file}: {e}")

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
    temp_files = []  # Keep track of all temporary files
    
    try:
        logger.info(f"Generating attack report for {original_path} to {output_path}")
        
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
        fig = None
        try:
            fig = plt.figure(figsize=(8, 4))
            
            # Use color coding for different classes
            colors = ['#2a9d8f', '#e63946', '#f4a261', '#457b9d']
            
            # Create the bar chart
            bars = plt.bar(classification_results['class_names'], 
                   [prob * 100 for prob in classification_results['probabilities']], 
                   color=colors)
                   
            # Add percentage labels above bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                         f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.xlabel('Attack Type')
            plt.ylabel('Confidence Percentage (%)')
            plt.title('Adversarial Image Classification Results')
            plt.ylim(0, max([prob * 100 for prob in classification_results['probabilities']]) + 10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save plot to a temporary file
            plot_file = save_plot_to_temp_file(fig)
            if plot_file:
                temp_files.append(plot_file)
                add_image_to_pdf(pdf, plot_file, w=170)
            plt.close(fig)  # Ensure the figure is closed
            fig = None
        except Exception as e:
            logger.error(f"Error generating probability chart: {str(e)}")
            pdf.cell(0, 10, f"Error generating probability chart: {str(e)}", 0, 1)
        finally:
            # Ensure figure is closed
            if fig is not None:
                plt.close(fig)
        
        # Add model contribution chart if available
        if 'attention_weights' in classification_results:
            add_model_contribution_chart(pdf, classification_results['attention_weights'], temp_files, "attack")
        
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
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create if there's a directory part
                os.makedirs(output_dir, exist_ok=True)
            pdf.output(output_path)
            logger.info(f"Successfully generated attack report at {output_path}")
        except Exception as e:
            logger.error(f"Error saving PDF report: {str(e)}")
            
        return output_path
    except Exception as e:
        logger.error(f"Error in generate_attack_report: {str(e)}")
        return None
    finally:
        # Clean up all temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file}: {e}")