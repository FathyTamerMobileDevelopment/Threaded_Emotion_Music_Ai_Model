import os
import sys
import argparse
from src.emotion_detector import EmotionDetector
from src.music_player import MusicPlayer
from src.ui_manager import UIManager
from utils.helpers import (
    create_project_structure,
    download_sample_model,
    get_sample_egyptian_music,
    check_tensorflow_installation,
    test_camera
)

def setup_environment():
    """
    Set up the initial environment, downloading models if needed
    """
    print("Setting up Emotion-Based Egyptian Music Player...")
        
    # Check TensorFlow installation
    tf_installed = check_tensorflow_installation()
    if not tf_installed:
        print("\nWarning: TensorFlow is not installed.")
        print("The application will try to continue, but may not function correctly.")
        print("Please install TensorFlow using the instructions above.\n")
    
    # Check for model file
    model_path = os.path.join("models", "emotion_model.h5")
    if not os.path.exists(model_path):
        print(f"Emotion model not found at {model_path}")
        success = download_sample_model(model_path)
        if not success:
            print("Warning: Could not download pre-trained model.")
            print("The application will download or create a simple model on startup.")
    
    # Test camera
    camera_working = test_camera()
    if not camera_working:
        print("\nWarning: Camera test failed.")
        print("Please check your webcam connection and permissions.\n")
    
    # Show music setup instructions
    get_sample_egyptian_music()
    
    print("\nSetup complete! The application is ready to start.")
    return True

def train_emotion_model(num_threads=None, use_gui=False):
    """
    Train the emotion detection model using the specified number of threads
    
    Args:
        num_threads: Number of threads to use for training (if None and not use_gui, user will be prompted)
        use_gui: If True, show a GUI dialog to prompt for thread count
    """
    print("Starting model training...")
    
    try:
        # Import the threaded trainer
        from threaded_emotion_training import ThreadedEmotionTrainer, prompt_for_threads
        
        # If no threads specified or GUI requested, prompt user
        if num_threads is None or use_gui:
            num_threads = prompt_for_threads()
        
        print(f"Using {num_threads} threads for training")
        
        # Initialize and run the trainer
        trainer = ThreadedEmotionTrainer(num_threads=num_threads)
        trainer.preprocess_and_train()
        
        print("Model training completed successfully!")
        return True
    except Exception as e:
        print(f"Error during model training: {e}")
        return False

def main():
    """
    Main function to run the Emotion-Based Egyptian Music Player
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Emotion-Based Egyptian Music Player')
    parser.add_argument('--train', action='store_true', help='Train the emotion detection model')
    parser.add_argument('--threads', type=int, help='Number of threads for training')
    parser.add_argument('--gui', action='store_true', help='Use GUI to prompt for thread count')
    args = parser.parse_args()
    
    # Create project structure if it doesn't exist
    create_project_structure()
    
    # If training mode is enabled, train the model and exit
    if args.train:
        success = train_emotion_model(args.threads, args.gui)
        if not success:
            print("Model training failed. Please check the logs.")
        return
    
    # Set up the environment
    setup_successful = setup_environment()
    if not setup_successful:
        print("Setup failed. Exiting...")
        return
    
    # Import components
    try:
        from src.emotion_detector import EmotionDetector
        from src.music_player import MusicPlayer
        from src.ui_manager import UIManager
    except ImportError as e:
        print(f"Error: Could not import application modules: {e}")
        print("Make sure the src folder contains the required modules.")
        return
    
    # Create application components
    try:
        print("\nInitializing components...")
        
        # Initialize emotion detector
        model_path = os.path.join("models", "emotion_model.h5")
        detector = EmotionDetector(model_path=model_path if os.path.exists(model_path) else None)
        
        # Initialize music player
        music_dir = os.path.join("data", "music")
        player = MusicPlayer(music_root=music_dir)
        
        # Initialize UI
        ui = UIManager(detector, player)
        
        print("Starting application...\n")
        
        # Start the UI
        ui.run()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()