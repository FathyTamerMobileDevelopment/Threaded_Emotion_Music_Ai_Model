import os
import random
import urllib.request
import zipfile
import io
import shutil

def create_project_structure():
    """
    Create the initial project folder structure
    """
    directories = [
        "models",
        "data/music/happy",
        "data/music/sad",
        "data/music/neutral",
        "data/music/angry",
        "data/music/surprised",
        "src",
        "utils"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    #عاملو احتياطي عشان لو ال مودل بتاع ال  ترينيج باظ 
def download_sample_model(save_path):
    """
    Download a pre-trained emotion recognition model
    
    Note: In a production system, you would want to use a proper emotion recognition model
          This function is a placeholder for demonstration purposes.
    """
    model_url = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
    
    try:
        print(f"Downloading emotion model from {model_url}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Download the file
        urllib.request.urlretrieve(model_url, save_path)
        print(f"Downloaded model to {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def get_sample_egyptian_music():
    """
    Provides instructions on how to obtain Egyptian music for the application
    """
    print("\n===== FATHY TAMER =====")


def check_tensorflow_installation():
    """
    Check if TensorFlow is installed and guide the user if it's not
    """
    try:
        import tensorflow as tf
        print(f"TensorFlow is installed (version {tf.__version__})")
        return True
    except ImportError:
        print("\n===== FATHY TAMER =====")

        return False

def test_camera():
    """
    Test if the webcam is working properly
    """
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam!")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        
        # Release the camera
        cap.release()
        
        if ret:
            print("Camera test: Successful!")
            return True
        else:
            print("Camera test: Failed to capture frame!")
            return False
            
    except Exception as e:
        print(f"Camera test error: {e}")
        return False

# For testing
if __name__ == "__main__":
    print("Testing utility functions...")
    
    # Test project structure creation
    create_project_structure()
    
    # Check TensorFlow installation
    tf_installed = check_tensorflow_installation()
    
    # Test camera
    camera_working = test_camera()
    
    # Show music instructions
    get_sample_egyptian_music()
    
    print("\nSystem check complete:")
    print(f"- Project structure: Created")
    print(f"- TensorFlow: {'Installed' if tf_installed else 'Not installed'}")
    print(f"- Camera: {'Working' if camera_working else 'Not working'}")
    print(f"- Music: Manual setup required")