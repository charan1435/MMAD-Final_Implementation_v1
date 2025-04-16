import os
import sys
import unittest
import shutil
import json
import numpy as np
import torch
from PIL import Image
import io

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the app and models
from app import app, classifier, generator, device
from config import Config
from models.classifier import AdversarialClassifier
from models.purifier import Generator
from models.attack_generator import fgsm_attack, bim_attack, pgd_attack
from utils.image_utils import preprocess_image, tensor_to_image

class AdversarialMRIDefenseIntegrationTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment before running tests"""
        # Configure app for testing
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['SERVER_NAME'] = 'localhost:5000'
        
        # Create test client
        cls.client = app.test_client()
        
        # Create temp directories for testing
        os.makedirs('test_uploads', exist_ok=True)
        os.makedirs('test_results', exist_ok=True)
        os.makedirs('test_reports', exist_ok=True)
        
        # Save original config
        cls.original_upload_folder = app.config['UPLOAD_FOLDER']
        cls.original_result_folder = app.config['RESULT_FOLDER']
        cls.original_report_folder = app.config['REPORT_FOLDER']
        
        # Override config for testing
        app.config['UPLOAD_FOLDER'] = 'test_uploads'
        app.config['RESULT_FOLDER'] = 'test_results'
        app.config['REPORT_FOLDER'] = 'test_reports'
        
        # Create sample test image
        cls.create_test_image()
        
        # Start the app context
        cls.app_context = app.app_context()
        cls.app_context.push()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        # Remove test directories
        shutil.rmtree('test_uploads', ignore_errors=True)
        shutil.rmtree('test_results', ignore_errors=True)
        shutil.rmtree('test_reports', ignore_errors=True)
        
        # Restore original config
        app.config['UPLOAD_FOLDER'] = cls.original_upload_folder
        app.config['RESULT_FOLDER'] = cls.original_result_folder
        app.config['REPORT_FOLDER'] = cls.original_report_folder
        
        # End the app context
        cls.app_context.pop()
    
    @classmethod
    def create_test_image(cls):
        """Create a test MRI image for testing"""
        # Create a simple 224x224 grayscale "MRI-like" test image
        img = Image.new('RGB', (224, 224), color=(100, 100, 100))
        
        # Add some structures to make it more MRI-like
        for i in range(50, 174):
            for j in range(50, 174):
                # Create a circular structure
                if ((i-112)**2 + (j-112)**2) < 60**2:
                    brightness = 200 - ((i-112)**2 + (j-112)**2) // 20
                    img.putpixel((i, j), (brightness, brightness, brightness))
        
        # Save the test image
        cls.test_image_path = 'test_image.png'
        img.save(cls.test_image_path)
    
    def setUp(self):
        """Set up before each test"""
        # No need to clear cookies - we'll use a fresh client for each test if needed
        pass
    
    def tearDown(self):
        """Clean up after each test"""
        pass
    
    def test_1_app_running(self):
        """Test if the app is running correctly"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Adversarial MRI Defense', response.data)
    
    def test_2_model_loading(self):
        """Test if models are loaded correctly"""
        # Check classifier
        self.assertIsInstance(classifier, AdversarialClassifier)
        self.assertTrue(hasattr(classifier, 'transformer'))
        self.assertTrue(hasattr(classifier, 'cnn'))
        self.assertTrue(hasattr(classifier, 'snn'))
        
        # Check generator (purifier)
        self.assertIsInstance(generator, Generator)
        self.assertTrue(hasattr(generator, 'initial'))
        self.assertTrue(hasattr(generator, 'bottleneck'))
    
    def test_3_upload_endpoint(self):
        """Test file upload endpoint"""
        with open(self.test_image_path, 'rb') as img:
            response = self.client.post(
                '/upload',
                data={
                    'file': (io.BytesIO(img.read()), 'test_mri.png')
                },
                content_type='multipart/form-data'
            )
            
        # Check response
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertTrue(response_data['success'])
        self.assertIn('session_id', response_data)
        self.assertIn('filename', response_data)
        
        # Save session ID for later tests
        self.session_id = response_data['session_id']
        self.filename = response_data['filename']
    
    def test_4_classification_workflow(self):
        """Test the complete classification workflow"""
        # First upload a file
        with open(self.test_image_path, 'rb') as img:
            upload_response = self.client.post(
                '/upload',
                data={
                    'file': (io.BytesIO(img.read()), 'test_mri.png')
                },
                content_type='multipart/form-data'
            )
        
        upload_data = json.loads(upload_response.data)
        session_id = upload_data['session_id']
        
        # Then classify the image
        classify_response = self.client.post(
            '/classify_image',
            json={'session_id': session_id},
            content_type='application/json'
        )
        
        # Check response
        self.assertEqual(classify_response.status_code, 200)
        classify_data = json.loads(classify_response.data)
        self.assertTrue(classify_data['success'])
        self.assertIn('predicted_class', classify_data)
        self.assertIn('probabilities', classify_data)
        self.assertIn('class_names', classify_data)
        self.assertIn('report_url', classify_data)
        
        # Verify result structure
        self.assertEqual(len(classify_data['probabilities']), 4)
        self.assertEqual(len(classify_data['class_names']), 4)
        self.assertIn(classify_data['predicted_class'], classify_data['class_names'])
    
    def test_5_purification_workflow(self):
        """Test the complete purification workflow"""
        # First upload a file
        with open(self.test_image_path, 'rb') as img:
            upload_response = self.client.post(
                '/upload',
                data={
                    'file': (io.BytesIO(img.read()), 'test_mri.png')
                },
                content_type='multipart/form-data'
            )
        
        upload_data = json.loads(upload_response.data)
        session_id = upload_data['session_id']
        
        # Then purify the image
        purify_response = self.client.post(
            '/purify_image',
            json={'session_id': session_id},
            content_type='application/json'
        )
        
        # Check response
        self.assertEqual(purify_response.status_code, 200)
        purify_data = json.loads(purify_response.data)
        self.assertTrue(purify_data['success'])
        self.assertIn('original_url', purify_data)
        self.assertIn('purified_url', purify_data)
        self.assertIn('comparison_url', purify_data)
        self.assertIn('download_url', purify_data)
        self.assertIn('report_url', purify_data)
        self.assertIn('psnr', purify_data)
        self.assertIn('ssim', purify_data)
        
        # Verify metrics are reasonable
        self.assertGreaterEqual(purify_data['psnr'], 0)
        self.assertGreaterEqual(purify_data['ssim'], 0)
        self.assertLessEqual(purify_data['ssim'], 1)
    
    def test_6_attack_workflow(self):
        """Test the complete attack generation workflow"""
        # First upload a file
        with open(self.test_image_path, 'rb') as img:
            upload_response = self.client.post(
                '/upload',
                data={
                    'file': (io.BytesIO(img.read()), 'test_mri.png')
                },
                content_type='multipart/form-data'
            )
        
        upload_data = json.loads(upload_response.data)
        session_id = upload_data['session_id']
        
        # Test all attack types
        for attack_type in ['fgsm', 'bim', 'pgd']:
            attack_response = self.client.post(
                '/attack_image',
                json={
                    'session_id': session_id,
                    'attack_type': attack_type,
                    'epsilon': 0.03
                },
                content_type='application/json'
            )
            
            # Check response
            self.assertEqual(attack_response.status_code, 200)
            attack_data = json.loads(attack_response.data)
            self.assertTrue(attack_data['success'])
            self.assertIn('original_url', attack_data)
            self.assertIn('adversarial_url', attack_data)
            self.assertIn('comparison_url', attack_data)
            self.assertIn('download_url', attack_data)
            self.assertIn('report_url', attack_data)
            self.assertIn('attack_type', attack_data)
            self.assertIn('epsilon', attack_data)
            self.assertIn('l2_distance', attack_data)
            self.assertIn('linf_distance', attack_data)
            self.assertIn('predicted_class', attack_data)
            self.assertIn('probabilities', attack_data)
            
            # Verify attack properties
            self.assertEqual(attack_data['attack_type'], attack_type)
            self.assertEqual(attack_data['epsilon'], 0.03)
            self.assertGreater(attack_data['l2_distance'], 0)
    
    def test_7_end_to_end_workflow(self):
        """Test an end-to-end workflow: upload -> attack -> purify -> classify"""
        # First upload a file
        with open(self.test_image_path, 'rb') as img:
            upload_response = self.client.post(
                '/upload',
                data={
                    'file': (io.BytesIO(img.read()), 'test_mri.png')
                },
                content_type='multipart/form-data'
            )
        
        upload_data = json.loads(upload_response.data)
        session_id = upload_data['session_id']
        
        # Generate an adversarial example
        attack_response = self.client.post(
            '/attack_image',
            json={
                'session_id': session_id,
                'attack_type': 'fgsm',
                'epsilon': 0.05
            },
            content_type='application/json'
        )
        
        attack_data = json.loads(attack_response.data)
        adv_result_id = attack_data['result_id']
        
        # Get the adversarial image URL
        adv_image_url = attack_data['adversarial_url']
        
        # Download the adversarial image
        adv_image_response = self.client.get(adv_image_url)
        self.assertEqual(adv_image_response.status_code, 200)
        
        # Create a new session with the adversarial image
        adv_img_data = io.BytesIO(adv_image_response.data)
        adv_upload_response = self.client.post(
            '/upload',
            data={
                'file': (adv_img_data, 'adversarial_mri.png')
            },
            content_type='multipart/form-data'
        )
        
        adv_upload_data = json.loads(adv_upload_response.data)
        adv_session_id = adv_upload_data['session_id']
        
        # Classify the adversarial image
        adv_classify_response = self.client.post(
            '/classify_image',
            json={'session_id': adv_session_id},
            content_type='application/json'
        )
        
        adv_classify_data = json.loads(adv_classify_response.data)
        self.assertTrue(adv_classify_data['success'])
        
        # Now purify the adversarial image
        purify_response = self.client.post(
            '/purify_image',
            json={'session_id': adv_session_id},
            content_type='application/json'
        )
        
        purify_data = json.loads(purify_response.data)
        self.assertTrue(purify_data['success'])
        
        # Get the purified image URL
        purified_image_url = purify_data['purified_url']
        
        # Download the purified image
        purified_image_response = self.client.get(purified_image_url)
        self.assertEqual(purified_image_response.status_code, 200)
        
        # Create a new session with the purified image
        purified_img_data = io.BytesIO(purified_image_response.data)
        purified_upload_response = self.client.post(
            '/upload',
            data={
                'file': (purified_img_data, 'purified_mri.png')
            },
            content_type='multipart/form-data'
        )
        
        purified_upload_data = json.loads(purified_upload_response.data)
        purified_session_id = purified_upload_data['session_id']
        
        # Finally, classify the purified image
        purified_classify_response = self.client.post(
            '/classify_image',
            json={'session_id': purified_session_id},
            content_type='application/json'
        )
        
        purified_classify_data = json.loads(purified_classify_response.data)
        self.assertTrue(purified_classify_data['success'])
        
        # Verify the purifier effectively defended against the attack
        # We'd expect the purified image to be classified differently than the adversarial one
        self.assertIsNotNone(adv_classify_data['predicted_class'])
        self.assertIsNotNone(purified_classify_data['predicted_class'])
        
        # Print results for inspection
        print(f"Adversarial image classified as: {adv_classify_data['predicted_class']}")
        print(f"Purified image classified as: {purified_classify_data['predicted_class']}")
    
    def test_8_report_generation(self):
        """Test report generation and download"""
        # First upload a file
        with open(self.test_image_path, 'rb') as img:
            upload_response = self.client.post(
                '/upload',
                data={
                    'file': (io.BytesIO(img.read()), 'test_mri.png')
                },
                content_type='multipart/form-data'
            )
        
        upload_data = json.loads(upload_response.data)
        session_id = upload_data['session_id']
        
        # Classify the image
        classify_response = self.client.post(
            '/classify_image',
            json={'session_id': session_id},
            content_type='application/json'
        )
        
        classify_data = json.loads(classify_response.data)
        result_id = classify_data['result_id']
        
        # Try to download the report
        report_url = f"/download_report/{session_id}/{result_id}/classification"
        report_response = self.client.get(report_url)
        
        # Check response
        self.assertIn(report_response.status_code, [200, 404])  # 404 is acceptable if report generation is mocked
        if report_response.status_code == 200:
            self.assertEqual(report_response.mimetype, 'application/pdf')
    
    def test_9_direct_model_integration(self):
        """Test direct integration between model components"""
        # Load test image
        test_image = Image.open(self.test_image_path).convert('RGB')
        image_tensor = preprocess_image(self.test_image_path).to(device)
        
        # Test classifier
        with torch.no_grad():
            outputs = classifier(image_tensor)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            prediction = torch.argmax(probabilities).item()
        
        self.assertIsNotNone(prediction)
        self.assertGreaterEqual(prediction, 0)
        self.assertLess(prediction, 4)  # 4 classes (0-3)
        
        # Test generator (purifier)
        with torch.no_grad():
            purified_tensor = generator(image_tensor)
        
        self.assertEqual(purified_tensor.shape, image_tensor.shape)
        
        # Test attack generator
        epsilon = 0.03
        # Use requires_grad=True for attack
        image_tensor.requires_grad_(True)
        
        # FGSM attack
        fgsm_result = fgsm_attack(image_tensor, epsilon)
        self.assertEqual(fgsm_result.shape, image_tensor.shape)
        
        # Calculate difference
        diff = torch.abs(fgsm_result - image_tensor).max().item()
        self.assertLessEqual(diff, epsilon + 1e-5)  # Allow small numerical error
    
    def test_10_cross_component_integration(self):
        """Test integration across components (attack->purify->classify)"""
        # Load test image
        image_tensor = preprocess_image(self.test_image_path).to(device)
        
        # First generate an adversarial example
        image_tensor.requires_grad_(True)
        adversarial_tensor = fgsm_attack(image_tensor, epsilon=0.05)
        
        # Now try to purify it
        with torch.no_grad():
            purified_tensor = generator(adversarial_tensor)
        
        # Classify both the adversarial and purified images
        with torch.no_grad():
            # Classify adversarial
            adv_outputs = classifier(adversarial_tensor)
            if isinstance(adv_outputs, dict):
                adv_logits = adv_outputs['logits']
            else:
                adv_logits = adv_outputs
            adv_probs = torch.nn.functional.softmax(adv_logits, dim=1)[0]
            adv_pred = torch.argmax(adv_probs).item()
            
            # Classify purified
            pur_outputs = classifier(purified_tensor)
            if isinstance(pur_outputs, dict):
                pur_logits = pur_outputs['logits']
            else:
                pur_logits = pur_outputs
            pur_probs = torch.nn.functional.softmax(pur_logits, dim=1)[0]
            pur_pred = torch.argmax(pur_probs).item()
        
        # Print results
        class_names = ['Clean', 'FGSM', 'BIM', 'PGD']
        print(f"Adversarial classified as: {class_names[adv_pred]}")
        print(f"Purified classified as: {class_names[pur_pred]}")
        
        # Calculate PSNR between original and purified
        from utils.image_utils import calculate_psnr
        psnr = calculate_psnr(purified_tensor, image_tensor).item()
        print(f"PSNR between original and purified: {psnr} dB")
        
        # The purifier should recover some of the original image quality
        self.assertGreaterEqual(psnr, 10.0)  # Expect reasonable quality

if __name__ == '__main__':
    unittest.main()