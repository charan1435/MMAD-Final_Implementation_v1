import os
import sys
import time
import uuid
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename

# Import custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from utils.image_utils import preprocess_image, save_image, generate_grid_visualization
from utils.report_generator import generate_classification_report, generate_purification_report, generate_attack_report
from models.classifier import AdversarialClassifier
from models.purifier import Generator, PatchDiscriminator
from models.attack_generator import fgsm_attack, bim_attack, pgd_attack

app = Flask(__name__)
app.secret_key = 'adversarial-mri-defense-key'
app.config.from_object(Config)

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)
os.makedirs(os.path.dirname(app.config['CHECKPOINT_FOLDER']), exist_ok=True)
os.makedirs(os.path.join(app.config['CHECKPOINT_FOLDER'], 'classifier'), exist_ok=True)
os.makedirs(os.path.join(app.config['CHECKPOINT_FOLDER'], 'purifier'), exist_ok=True)

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load classifier model
classifier = AdversarialClassifier().to(device)
classifier_checkpoint_path = os.path.join(app.config['CHECKPOINT_FOLDER'], 'classifier', 'best_model.pth')
if os.path.exists(classifier_checkpoint_path):
    try:
        checkpoint = torch.load(classifier_checkpoint_path, map_location=device)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded classifier model from {classifier_checkpoint_path}")
    except Exception as e:
        print(f"Warning: Error loading classifier checkpoint: {e}")
        print("Running with initialized classifier model (not trained)")
else:
    print(f"Warning: Classifier checkpoint not found at {classifier_checkpoint_path}")
classifier.eval()

# Load purifier model - using phase2 model and weights_only=False
generator = Generator().to(device)
purifier_checkpoint_path = os.path.join(app.config['CHECKPOINT_FOLDER'], 'purifier', 'best_model_phase2.pth')
if os.path.exists(purifier_checkpoint_path):
    try:
        # Use weights_only=False to allow loading NumPy scalar values
        checkpoint = torch.load(purifier_checkpoint_path, map_location=device, weights_only=False)
        
        # Check which key contains the model weights
        if 'model_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['model_state_dict'])
        elif 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            # Try to load the checkpoint directly if it's just a state dict
            generator.load_state_dict(checkpoint)
            
        print(f"Loaded purifier model from {purifier_checkpoint_path}")
    except Exception as e:
        print(f"Warning: Error loading purifier checkpoint: {e}")
        print("Running with initialized purifier model (not trained)")
else:
    print(f"Warning: Purifier checkpoint not found at {purifier_checkpoint_path}")
generator.eval()

# Initialize transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Model for session handling
class Session:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.timestamp = time.time()
        self.upload_filename = None
        self.upload_path = None
        self.results = {}
        
sessions = {}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify')
def classify():
    return render_template('classify.html')

@app.route('/purify')
def purify():
    return render_template('purify.html')

@app.route('/attack')
def attack():
    return render_template('attack.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Create new session
        session = Session()
        sessions[session.session_id] = session
        
        # Save original filename and path
        filename = secure_filename(file.filename)
        session.upload_filename = filename
        
        # Create directory for this session
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session.session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save file
        filepath = os.path.join(session_dir, filename)
        file.save(filepath)
        session.upload_path = filepath
        
        return jsonify({
            'success': True,
            'session_id': session.session_id,
            'filename': filename,
            'message': 'File uploaded successfully'
        })
    
    return jsonify({
        'success': False,
        'message': 'Invalid file type'
    })

@app.route('/classify_image', methods=['POST'])
def classify_image():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({
            'success': False,
            'message': 'Invalid session'
        })
    
    session = sessions[session_id]
    
    try:
        # Check if the uploaded file exists
        if not os.path.exists(session.upload_path):
            return jsonify({
                'success': False,
                'message': 'Uploaded file not found. Please upload again.'
            })
        
        # Load and preprocess image
        image = Image.open(session.upload_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Perform classification
        with torch.no_grad():
            # Get full outputs including attention weights
            outputs = classifier(image_tensor)
            
            # Check if the output is a dictionary (training mode) or tensor (eval mode)
            if isinstance(outputs, dict):
                logits = outputs['logits']
                attention_weights = outputs.get('attention_weights', None)
            else:
                logits = outputs
                attention_weights = None
                
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            prediction = torch.argmax(probabilities).item()
            
        # Map prediction to class
        class_names = ['Clean', 'FGSM', 'BIM', 'PGD']
        predicted_class = class_names[prediction]
        
        # Get probability scores
        probs = probabilities.cpu().numpy().tolist()
        
        # Save results to session
        result_id = str(uuid.uuid4())
        result_data = {
            'operation': 'classification',
            'predicted_class': predicted_class,
            'probabilities': probs,
            'class_names': class_names,
            'timestamp': time.time()
        }
        
        # Add attention weights if available
        if attention_weights is not None:
            result_data['attention_weights'] = attention_weights.cpu().numpy().tolist()
        
        session.results[result_id] = result_data
        
        # Create result directory
        report_dir = os.path.join(app.config['REPORT_FOLDER'])
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate report
        report_path = os.path.join(report_dir, f"{result_id}_classification_report.pdf")
        report_url = f"/download_report/{session_id}/{result_id}/classification"
        
        report_data = {
            'filename': session.upload_filename,
            'predicted_class': predicted_class,
            'probabilities': probs,
            'class_names': class_names,
            'result_id': result_id,
            'report_url': report_url
        }
        
        # Add attention weights to report data if available
        if attention_weights is not None:
            report_data['attention_weights'] = attention_weights.cpu().numpy().tolist()
        
        # Generate report with error handling
        try:
            generate_classification_report(
                image_path=session.upload_path,
                result_data=report_data,
                output_path=report_path
            )
        except Exception as e:
            print(f"Warning: Error generating classification report: {str(e)}")
            # Continue without the report - we'll return success even if the report fails
        
        return jsonify({
            'success': True,
            'result_id': result_id,
            'predicted_class': predicted_class,
            'probabilities': probs,
            'class_names': class_names,
            'report_url': report_url
        })
        
    except Exception as e:
        print(f"Error in classify_image: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error processing image: {str(e)}'
        })

@app.route('/purify_image', methods=['POST'])
def purify_image():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({
            'success': False,
            'message': 'Invalid session'
        })
    
    session = sessions[session_id]
    
    try:
        # Check if the uploaded file exists
        if not os.path.exists(session.upload_path):
            return jsonify({
                'success': False,
                'message': 'Uploaded file not found. Please upload again.'
            })
            
        # Load and preprocess image
        image = Image.open(session.upload_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate purified image
        with torch.no_grad():
            purified_tensor = generator(image_tensor)
            
        # Create result directory for this session
        session_dir = os.path.join(app.config['RESULT_FOLDER'], session.session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save purified image
        result_id = str(uuid.uuid4())
        purified_filename = f"purified_{result_id}_{session.upload_filename}"
        purified_path = os.path.join(session_dir, purified_filename)
        
        # Convert tensor to image and save
        purified_image = transforms.ToPILImage()(purified_tensor.squeeze().cpu())
        purified_image.save(purified_path)
        
        # Create comparison visualization
        comparison_filename = f"comparison_{result_id}_{session.upload_filename}"
        comparison_path = os.path.join(session_dir, comparison_filename)
        
        # Generate side-by-side comparison
        generate_grid_visualization(
            [image, purified_image],
            ["Original", "Purified"],
            comparison_path
        )
        
        # Calculate PSNR and SSIM metrics
        from utils.image_utils import calculate_psnr, calculate_ssim
        
        original_tensor = image_tensor.to(device)
        psnr_value = calculate_psnr(purified_tensor, original_tensor).item()
        ssim_value = calculate_ssim(purified_tensor, original_tensor).item()
        
        # Save results to session
        session.results[result_id] = {
            'operation': 'purification',
            'original_path': session.upload_path,
            'purified_path': purified_path,
            'comparison_path': comparison_path,
            'psnr': psnr_value,
            'ssim': ssim_value,
            'timestamp': time.time()
        }
        
        # Generate report
        report_dir = os.path.join(app.config['REPORT_FOLDER'])
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, f"{result_id}_purification_report.pdf")
        report_url = f"/download_report/{session_id}/{result_id}/purification"
        
        # Generate purification report with error handling
        try:
            generate_purification_report(
                original_path=session.upload_path,
                purified_path=purified_path,
                comparison_path=comparison_path,
                metrics={'psnr': psnr_value, 'ssim': ssim_value},
                output_path=report_path
            )
        except Exception as e:
            print(f"Warning: Error generating purification report: {str(e)}")
            # Continue without the report
        
        return jsonify({
            'success': True,
            'result_id': result_id,
            'original_url': f"/get_image/{session_id}/{session.upload_filename}",
            'purified_url': f"/get_result/{session_id}/{purified_filename}",
            'comparison_url': f"/get_result/{session_id}/{comparison_filename}",
            'download_url': f"/download_image/{session_id}/{result_id}",
            'report_url': report_url,
            'psnr': psnr_value,
            'ssim': ssim_value
        })
        
    except Exception as e:
        print(f"Error in purify_image: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error processing image: {str(e)}'
        })

@app.route('/attack_image', methods=['POST'])
def attack_image():
    data = request.json
    session_id = data.get('session_id')
    attack_type = data.get('attack_type', 'fgsm')
    epsilon = float(data.get('epsilon', 0.03))
    
    if session_id not in sessions:
        return jsonify({
            'success': False,
            'message': 'Invalid session'
        })
    
    session = sessions[session_id]
    
    # Validate parameters
    if attack_type not in ['fgsm', 'bim', 'pgd']:
        return jsonify({
            'success': False,
            'message': 'Invalid attack type'
        })
    
    if epsilon <= 0 or epsilon > 0.1:
        return jsonify({
            'success': False,
            'message': 'Epsilon must be between 0 and 0.1'
        })
    
    try:
        # Check if the uploaded file exists
        if not os.path.exists(session.upload_path):
            return jsonify({
                'success': False,
                'message': 'Uploaded file not found. Please upload again.'
            })
            
        # Load and preprocess image
        image = Image.open(session.upload_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        image_tensor.requires_grad = True
        
        # Perform the chosen attack
        if attack_type == 'fgsm':
            adversarial_tensor = fgsm_attack(image_tensor, epsilon)
        elif attack_type == 'bim':
            adversarial_tensor = bim_attack(image_tensor, epsilon, alpha=epsilon/10, iterations=10)
        elif attack_type == 'pgd':
            adversarial_tensor = pgd_attack(image_tensor, epsilon, alpha=epsilon/10, iterations=20)
        
        # Create result directory for this session
        session_dir = os.path.join(app.config['RESULT_FOLDER'], session.session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save adversarial image
        result_id = str(uuid.uuid4())
        adversarial_filename = f"{attack_type}_{result_id}_{session.upload_filename}"
        adversarial_path = os.path.join(session_dir, adversarial_filename)
        
        # Convert tensor to image and save
        adversarial_image = transforms.ToPILImage()(adversarial_tensor.squeeze().cpu().detach())
        adversarial_image.save(adversarial_path)
        
        # Create comparison visualization
        comparison_filename = f"comparison_{attack_type}_{result_id}_{session.upload_filename}"
        comparison_path = os.path.join(session_dir, comparison_filename)
        
        # Generate side-by-side comparison
        generate_grid_visualization(
            [image, adversarial_image],
            ["Original", f"{attack_type.upper()} (ε={epsilon})"],
            comparison_path
        )
        
        # Calculate perturbation metrics
        l2_distance = torch.norm(adversarial_tensor - image_tensor, p=2).item()
        linf_distance = torch.norm(adversarial_tensor - image_tensor, p=float('inf')).item()
        
        # Also classify the adversarial image
        with torch.no_grad():
            # Get full outputs including attention weights
            adv_outputs = classifier(adversarial_tensor)
            
            # Check if the output is a dictionary (training mode) or tensor (eval mode)
            if isinstance(adv_outputs, dict):
                adv_logits = adv_outputs['logits']
                attention_weights = adv_outputs.get('attention_weights', None)
            else:
                adv_logits = adv_outputs
                attention_weights = None
                
            adv_probabilities = torch.nn.functional.softmax(adv_logits, dim=1)[0]
            adv_prediction = torch.argmax(adv_probabilities).item()
            
        # Map prediction to class
        class_names = ['Clean', 'FGSM', 'BIM', 'PGD']
        adv_predicted_class = class_names[adv_prediction]
        adv_probs = adv_probabilities.cpu().numpy().tolist()
        
        # Prepare classification results for report
        classification_results = {
            'predicted_class': adv_predicted_class,
            'probabilities': adv_probs,
            'class_names': class_names
        }
        
        # Add attention weights if available
        if attention_weights is not None:
            classification_results['attention_weights'] = attention_weights.cpu().numpy().tolist()
        
        # Save results to session
        session.results[result_id] = {
            'operation': 'attack',
            'attack_type': attack_type,
            'epsilon': epsilon,
            'original_path': session.upload_path,
            'adversarial_path': adversarial_path,
            'comparison_path': comparison_path,
            'l2_distance': l2_distance,
            'linf_distance': linf_distance,
            'predicted_class': adv_predicted_class,
            'probabilities': adv_probs,
            'timestamp': time.time()
        }
        
        # Generate report path and URL
        report_dir = os.path.join(app.config['REPORT_FOLDER'])
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, f"{result_id}_attack_report.pdf")
        report_url = f"/download_report/{session_id}/{result_id}/attack"
        
        # Generate attack report with error handling
        try:
            generate_attack_report(
                original_path=session.upload_path,
                adversarial_path=adversarial_path,
                comparison_path=comparison_path,
                attack_params={'attack_type': attack_type, 'epsilon': epsilon},
                metrics={'l2_distance': l2_distance, 'linf_distance': linf_distance},
                classification_results=classification_results,
                output_path=report_path
            )
        except Exception as e:
            print(f"Warning: Error generating attack report: {str(e)}")
            # Continue without the report
        
        return jsonify({
            'success': True,
            'result_id': result_id,
            'original_url': f"/get_image/{session_id}/{session.upload_filename}",
            'adversarial_url': f"/get_result/{session_id}/{adversarial_filename}",
            'comparison_url': f"/get_result/{session_id}/{comparison_filename}",
            'download_url': f"/download_image/{session_id}/{result_id}/adversarial",
            'report_url': report_url,
            'attack_type': attack_type,
            'epsilon': epsilon,
            'l2_distance': l2_distance,
            'linf_distance': linf_distance,
            'predicted_class': adv_predicted_class,
            'probabilities': adv_probs,
            'class_names': class_names
        })
        
    except Exception as e:
        print(f"Error in attack_image: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error processing image: {str(e)}'
        })

@app.route('/get_image/<session_id>/<filename>')
def get_image(session_id, filename):
    if session_id not in sessions:
        flash('Invalid session')
        return redirect(url_for('index'))
    
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], session_id, filename)
    
    if not os.path.exists(image_path):
        flash('Image not found')
        return redirect(url_for('index'))
        
    return send_file(image_path)

@app.route('/get_result/<session_id>/<filename>')
def get_result(session_id, filename):
    if session_id not in sessions:
        flash('Invalid session')
        return redirect(url_for('index'))
    
    result_path = os.path.join(app.config['RESULT_FOLDER'], session_id, filename)
    
    if not os.path.exists(result_path):
        flash('Result not found')
        return redirect(url_for('index'))
        
    return send_file(result_path)

@app.route('/download_image/<session_id>/<result_id>')
@app.route('/download_image/<session_id>/<result_id>/<image_type>')
def download_image(session_id, result_id, image_type=None):
    if session_id not in sessions:
        flash('Invalid session')
        return redirect(url_for('index'))
    
    session = sessions[session_id]
    
    if result_id not in session.results:
        flash('Result not found')
        return redirect(url_for('index'))
    
    result = session.results[result_id]
    
    if result['operation'] == 'purification':
        if not os.path.exists(result['purified_path']):
            flash('Purified image not found')
            return redirect(url_for('index'))
            
        return send_file(result['purified_path'], as_attachment=True, 
                         download_name=f"purified_{os.path.basename(result['purified_path'])}")
    elif result['operation'] == 'attack':
        if not os.path.exists(result['adversarial_path']):
            flash('Adversarial image not found')
            return redirect(url_for('index'))
            
        return send_file(result['adversarial_path'], as_attachment=True, 
                         download_name=f"{result['attack_type']}_{os.path.basename(result['adversarial_path'])}")
    
    flash('Invalid operation')
    return redirect(url_for('index'))

@app.route('/download_report/<session_id>/<result_id>/<report_type>')
def download_report(session_id, result_id, report_type):
    if session_id not in sessions:
        flash('Invalid session')
        return redirect(url_for('index'))
    
    report_path = os.path.join(app.config['REPORT_FOLDER'], f"{result_id}_{report_type}_report.pdf")
    
    if not os.path.exists(report_path):
        flash('Report not found')
        return redirect(url_for('index'))
    
    return send_file(report_path, as_attachment=True, 
                     download_name=f"{report_type}_report_{result_id}.pdf")

# Clean up old sessions
@app.before_request
def cleanup_old_sessions():
    current_time = time.time()
    sessions_to_remove = []
    
    for session_id, session in sessions.items():
        # Remove sessions older than 2 hours
        if current_time - session.timestamp > 7200:
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        del sessions[session_id]

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)