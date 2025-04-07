import os

class Config:
    # Base directory of the application
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Folders
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    RESULT_FOLDER = os.path.join(BASE_DIR, 'results')
    REPORT_FOLDER = os.path.join(BASE_DIR, 'reports')
    CHECKPOINT_FOLDER = os.path.join(BASE_DIR, 'checkpoints')
    
    # MRI dataset paths (for reference)
    CLEAN_DIR = os.path.join(BASE_DIR, 'datasets', 'clean')
    ATTACK_DIRS = {
        'fgsm': os.path.join(BASE_DIR, 'datasets', 'fgsm'),
        'bim': os.path.join(BASE_DIR, 'datasets', 'bim'),
        'pgd': os.path.join(BASE_DIR, 'datasets', 'pgd')
    }
    
    # Model parameters
    BATCH_SIZE = 8
    INPUT_SIZE = 224
    IMG_CHANNELS = 3
    DEVICE = 'cuda' if os.path.exists('/dev/nvidia0') else 'cpu'
    
    # Flask configuration
    DEBUG = True
    SECRET_KEY = 'adversarial-mri-defense-secret-key'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Application name and version
    APP_NAME = 'Adversarial MRI Defense'
    APP_VERSION = '1.0.0'